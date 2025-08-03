import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Dict, Any
import math
from collections import defaultdict


class H2OKVCache:
    def __init__(
            self,
            model_name_or_path: str = "/home/liangtao/Models/Qwen/Qwen2-0.5B",
            h2o_ratio: float = 0.1,
            recent_ratio: float = 0.1,
            cache_size: int = 1024,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize H2O KV Cache manager with Qwen2 compatibility.
        """
        self.model_name_or_path = model_name_or_path
        self.h2o_ratio = h2o_ratio
        self.recent_ratio = recent_ratio
        self.cache_size = cache_size
        self.device = device

        # Load model with correct attention implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            attn_implementation="eager"  # Force eager attention for attention scores
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize cache
        self.kv_cache = None
        self.attention_scores = []
        self.token_positions = []
        self.token_attention_accumulator = defaultdict(float)
        self.current_seq_len = 0

    def reset_cache(self):
        """Reset the KV cache and tracking variables"""
        self.kv_cache = None
        self.attention_scores = []
        self.token_positions = []
        self.token_attention_accumulator = defaultdict(float)
        self.current_seq_len = 0

    def update_attention_tracking(self, attention_scores: List[torch.Tensor], input_ids: torch.Tensor):
        """
        Update attention tracking for heavy hitter identification.
        Handles Qwen2's attention scores format.
        """
        if not attention_scores:
            return

        last_layer_attention = attention_scores[-1]  # Get last layer attention

        # Handle different possible shapes
        if last_layer_attention.dim() == 3:
            # Shape: (batch_size, seq_len, seq_len)
            batch_size, seq_len, _ = last_layer_attention.shape
            num_heads = 1
            last_layer_attention = last_layer_attention.unsqueeze(1)  # Add head dimension
        elif last_layer_attention.dim() == 4:
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = last_layer_attention.shape
        else:
            raise ValueError(f"Unexpected attention scores shape: {last_layer_attention.shape}")

        # We only track the last token's attention to previous tokens
        last_token_attention = last_layer_attention[:, :, -1, :-1]  # [batch, heads, seq_len-1]
        last_token_attention = last_token_attention.mean(dim=(0, 1))  # Average over batch and heads

        # Get the corresponding token IDs
        token_ids = input_ids[0, :-1].cpu().numpy()

        # Update attention accumulator
        for token_id, score in zip(token_ids, last_token_attention):
            self.token_attention_accumulator[token_id] += score.item()

        # Store positions and scores for recent tokens
        self.current_seq_len += seq_len - (1 if self.current_seq_len > 0 else 0)

    def select_cache_indices(self) -> List[int]:
        """
        Select which tokens to keep in the cache based on H2O strategy.
        Ensures indices are within valid bounds.
        """
        if not self.token_attention_accumulator or self.current_seq_len == 0:
            return list(range(min(self.cache_size, self.current_seq_len)))

        total_keep = min(self.cache_size, self.current_seq_len)
        h2o_keep = int(total_keep * self.h2o_ratio)
        recent_keep = int(total_keep * self.recent_ratio)
        random_keep = max(0, total_keep - h2o_keep - recent_keep)  # Ensure non-negative

        # Select heavy hitters
        sorted_tokens = sorted(
            self.token_attention_accumulator.items(),
            key=lambda x: x[1],
            reverse=True
        )
        h2o_token_ids = {token_id for token_id, _ in sorted_tokens[:h2o_keep]}

        # Get positions of heavy hitters (ensure they're within current sequence length)
        h2o_indices = [
                          idx for idx, token_id in enumerate(self.token_positions)
                          if token_id in h2o_token_ids and idx < self.current_seq_len
                      ][:h2o_keep]  # Ensure we don't exceed the allocation

        # Select recent tokens (ensure they're within bounds)
        recent_start = max(0, self.current_seq_len - recent_keep)
        recent_indices = list(range(recent_start, min(self.current_seq_len, recent_start + recent_keep)))

        # Select random tokens for the rest (from remaining valid indices)
        remaining_indices = [
            i for i in range(self.current_seq_len)
            if i not in h2o_indices and i not in recent_indices
        ]
        random_indices = torch.randperm(len(remaining_indices))[:random_keep].tolist()
        random_indices = [remaining_indices[i] for i in random_indices if i < len(remaining_indices)]

        # Combine all selected indices and ensure they're unique and sorted
        selected_indices = list(set(h2o_indices + recent_indices + random_indices))
        selected_indices = [idx for idx in selected_indices if idx < self.current_seq_len]
        selected_indices.sort()

        # If we still don't have enough indices, fill with recent tokens
        if len(selected_indices) < total_keep:
            additional_needed = total_keep - len(selected_indices)
            additional_indices = list(range(max(0, self.current_seq_len - additional_needed), self.current_seq_len))
            selected_indices.extend(additional_indices)
            selected_indices = list(set(selected_indices))  # Remove duplicates
            selected_indices.sort()

        return selected_indices[:total_keep]  # Final safety check

    def create_cache_wrapper(self, kv_cache: Tuple) -> Any:
        """
        Create a cache wrapper that implements Qwen2's cache interface.
        """

        class CacheWrapper:
            def __init__(self, cache):
                self.cache = cache
                self._seen_tokens = 0

            def __getitem__(self, index):
                return self.cache[index]

            def __len__(self):
                return len(self.cache)

            def get_seq_length(self):
                if not self.cache or not self.cache[0] or self.cache[0][0] is None:
                    return self._seen_tokens
                # Assuming the first layer's key tensor has shape [batch, heads, seq_len, dim]
                return self.cache[0][0].shape[2] + self._seen_tokens

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                """
                Update the cache with new key and value states for a specific layer.
                """
                # Create new cache if it doesn't exist
                if self.cache is None:
                    num_layers = len(self.model.chat_model.layers) if hasattr(self.model, 'model') else len(
                        self.model.base_model.layers)
                    self.cache = tuple([(None, None) for _ in range(num_layers)])

                # Update the specified layer
                new_cache = list(self.cache)
                new_cache[layer_idx] = (key_states, value_states)
                self.cache = tuple(new_cache)
                self._seen_tokens += key_states.shape[2]  # Track total seen tokens
                return self.cache[layer_idx]

        return CacheWrapper(kv_cache)

    def sparse_kv_cache(self, kv_cache: Tuple, selected_indices: List[int]) -> Any:
        """
        Create a sparse KV cache by selecting only the specified indices.
        Returns a cache wrapper that Qwen2 can use.
        """
        if kv_cache is None or not selected_indices:
            return self.create_cache_wrapper(None)

        new_kv_cache = []
        for layer in kv_cache:
            key, value = layer
            if key is not None and value is not None:
                # Ensure indices are within bounds
                valid_indices = [idx for idx in selected_indices if idx < key.shape[2]]
                if not valid_indices:
                    # If no valid indices, keep at least one recent token
                    valid_indices = [min(key.shape[2] - 1, len(selected_indices) - 1)]

                # Convert indices to tensor and move to correct device
                indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=key.device)

                # Index select with valid indices
                new_key = key.index_select(2, indices_tensor)
                new_value = value.index_select(2, indices_tensor)
                new_kv_cache.append((new_key, new_value))
            else:
                new_kv_cache.append((None, None))

        return self.create_cache_wrapper(tuple(new_kv_cache))

    def generate(
            self,
            prompt: str,
            max_length: int = 100,
            temperature: float = 0.7,
            top_k: int = 50,
            top_p: float = 0.9,
            do_sample: bool = True
    ) -> str:
        """
        Generate text using the model with H2O KV cache management.
        """
        self.reset_cache()

        # Encode input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids

        for _ in range(max_length):
            # Forward pass with output_attentions
            outputs = self.model(
                input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
                output_attentions=True
            )

            # Update attention tracking
            attention_scores = outputs.attentions  # This is a tuple of attention tensors
            self.update_attention_tracking(attention_scores, input_ids)

            # Update token positions
            new_token_positions = input_ids[0].cpu().numpy().tolist()
            self.token_positions.extend(new_token_positions if not self.token_positions else new_token_positions[1:])

            # Get next token
            logits = outputs.logits[:, -1, :] / temperature
            if top_k > 0:
                logits = self.top_k_logits(logits, top_k)
            if top_p > 0:
                logits = self.top_p_logits(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            input_ids = next_token

            # Update KV cache with H2O strategy
            selected_indices = self.select_cache_indices()
            self.kv_cache = self.sparse_kv_cache(outputs.past_key_values, selected_indices)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    @staticmethod
    def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out

    @staticmethod
    def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float('Inf')
        return logits


if __name__ == "__main__":
    h2o_cache = H2OKVCache(
        model_name_or_path="/home/liangtao/Models/Qwen/Qwen2-0.5B",
        h2o_ratio=0.2,
        recent_ratio=0.3,
        cache_size=512,
        device="cuda"
    )

    prompt = "The future of artificial intelligence is"
    generated_text = h2o_cache.generate(
        prompt,
        max_length=200,
        temperature=0.2,
        top_k=50,
        top_p=0.9
    )

    print("Generated Text:")
    print(generated_text)