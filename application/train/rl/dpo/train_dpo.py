# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: train_dpo.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/13 21:02
# https://chat.deepseek.com/a/chat/s/ed3ca0e8-0012-4017-a3cc-115ff9872e67
# https://chat.deepseek.com/a/chat/s/369441e3-7a8e-4120-b81e-a4f8acf56ee1

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
import rdkit
from rdkit import Chem


@dataclass
class DPOSample:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_reward: float
    rejected_reward: float


class DPODataset(Dataset):
    def __init__(self, data: List[DPOSample], tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Tokenize chosen response
        chosen_text = sample.prompt + sample.chosen_response
        chosen_encodings = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        # Tokenize rejected response
        rejected_text = sample.prompt + sample.rejected_response
        rejected_encodings = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
            "chosen_reward": sample.chosen_reward,
            "rejected_reward": sample.rejected_reward
        }


def collate_fn(batch: List[Dict], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]

    # Pad sequences
    chosen_encodings = tokenizer.pad(
        {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask},
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    rejected_encodings = tokenizer.pad(
        {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask},
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    chosen_rewards = torch.tensor([item["chosen_reward"] for item in batch], dtype=torch.float32)
    rejected_rewards = torch.tensor([item["rejected_reward"] for item in batch], dtype=torch.float32)

    return {
        "chosen_input_ids": chosen_encodings["input_ids"],
        "chosen_attention_mask": chosen_encodings["attention_mask"],
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards
    }


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def smiles_equals(smiles1: str, smiles2: str) -> bool:
    """Check if two SMILES strings represent the same molecule"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return False
        return Chem.MolToInchiKey(mol1) == Chem.MolToInchiKey(mol2)
    except:
        return False


def calculate_reward(generated_smiles: str, target_smiles: str) -> float:
    """Calculate reward based on generated SMILES and target SMILES"""
    if smiles_equals(generated_smiles, target_smiles):
        return 1.0
    elif is_valid_smiles(generated_smiles):
        return 0.1
    else:
        return 0.0


def prepare_dpo_data(
        data_path: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        num_samples: int = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
) -> List[DPOSample]:
    """Prepare DPO data from JSON file"""
    with open(data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    if num_samples is not None:
        original_data = original_data[:num_samples]

    dpo_samples = []

    for item in tqdm(original_data, desc="Preparing DPO data"):
        prompt = item["instruction"] + "\n" + item["input"] + "\n"
        target_response = item["output"]

        # Generate rejected response using current model
        messages = [{"role": "user", "content": prompt.strip()}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_response = tokenizer.decode(
            outputs[0][len(input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        # Calculate rewards
        chosen_reward = calculate_reward(target_response, target_response)  # Should be 1.0
        rejected_reward = calculate_reward(generated_response, target_response)

        dpo_samples.append(DPOSample(
            prompt=prompt,
            chosen_response=target_response,
            rejected_response=generated_response,
            chosen_reward=chosen_reward,
            rejected_reward=rejected_reward
        ))

    return dpo_samples


def load_or_generate_dpo_data(
        data_path: str,
        dpo_data_path: str,
        dpo_data_file: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        num_samples: int = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
) -> List[DPOSample]:
    """Load DPO data if exists, otherwise generate and save it"""
    # Create DPO data file name
    dpo_filename = os.path.splitext(dpo_data_file)[0] + "_dpo.json"
    dpo_file_path = os.path.join(dpo_data_path, dpo_filename)

    # Check if DPO data already exists
    if os.path.exists(dpo_file_path):
        print(f"Loading existing DPO data from {dpo_file_path}")
        with open(dpo_file_path, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)

        # Convert dict back to DPOSample objects
        dpo_samples = []
        for item in dpo_data:
            dpo_samples.append(DPOSample(
                prompt=item["prompt"],
                chosen_response=item["chosen_response"],
                rejected_response=item["rejected_response"],
                chosen_reward=item["chosen_reward"],
                rejected_reward=item["rejected_reward"]
            ))
        return dpo_samples

    # Generate new DPO data
    print(f"Generating new DPO data and saving to {dpo_file_path}")
    data_file_path = os.path.join(data_path, dpo_data_file)
    dpo_samples = prepare_dpo_data(
        data_file_path,
        tokenizer,
        model,
        num_samples,
        max_length,
        temperature,
        top_p
    )

    # Convert DPOSample objects to dict for serialization
    dpo_data_dict = []
    for sample in dpo_samples:
        dpo_data_dict.append({
            "prompt": sample.prompt,
            "chosen_response": sample.chosen_response,
            "rejected_response": sample.rejected_response,
            "chosen_reward": sample.chosen_reward,
            "rejected_reward": sample.rejected_reward
        })

    # Save DPO data
    os.makedirs(dpo_data_path, exist_ok=True)
    with open(dpo_file_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data_dict, f, indent=2, ensure_ascii=False)

    return dpo_samples


def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DPO loss"""
    # Detach reference logprobs to prevent gradient flow
    reference_chosen_logps = reference_chosen_logps.detach()
    reference_rejected_logps = reference_rejected_logps.detach()

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    logits = policy_logratios - reference_logratios
    losses = -torch.nn.functional.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute log probabilities for given sequences"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift to align logits with labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens
    per_token_logps = per_token_logps * shift_attention_mask
    logps = per_token_logps.sum(dim=-1) / shift_attention_mask.sum(dim=-1)

    return logps


def train_dpo(
        model_path: str,
        dpo_data_path: str,
        dpo_data_file: str,
        output_path: str,
        log_path: str = None,
        num_samples: int = None,
        max_length: int = 512,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        beta: float = 0.1,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        checkpoint_steps: int = 500,
        seed: int = 42
):
    """Main DPO training function"""

    # Set seed
    set_seed(seed)

    # Setup logging
    if log_path is None:
        log_path = os.path.join(output_path, "training.log")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reference model (frozen)
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Load policy model (trainable)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    # Load or generate DPO data
    logger.info("Loading or generating DPO data...")
    dpo_data = load_or_generate_dpo_data(
        dpo_data_path,
        dpo_data_path,
        dpo_data_file,
        tokenizer,
        policy_model,
        num_samples,
        max_length
    )

    # Create dataset and dataloader
    dataset = DPODataset(dpo_data, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_length)
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=learning_rate
    )

    # Training loop
    logger.info("Starting DPO training...")
    global_step = 0
    policy_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # Move batch to device
            chosen_input_ids = batch["chosen_input_ids"].to(policy_model.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(policy_model.device)
            rejected_input_ids = batch["rejected_input_ids"].to(policy_model.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(policy_model.device)

            # Compute log probabilities for policy model
            policy_chosen_logps = compute_log_probs(policy_model, chosen_input_ids, chosen_attention_mask)
            policy_rejected_logps = compute_log_probs(policy_model, rejected_input_ids, rejected_attention_mask)

            # Compute log probabilities for reference model
            with torch.no_grad():
                reference_chosen_logps = compute_log_probs(reference_model, chosen_input_ids, chosen_attention_mask)
                reference_rejected_logps = compute_log_probs(reference_model, rejected_input_ids,
                                                             rejected_attention_mask)

            # Compute DPO loss
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress
            epoch_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / global_step:.4f}'
            })

            # Save checkpoint
            if global_step % checkpoint_steps == 0:
                checkpoint_dir = os.path.join(output_path, f"checkpoint-{global_step}")
                policy_model.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")

        logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(dataloader):.4f}")

    # Save final model
    final_model_dir = os.path.join(output_path, "final_model")
    policy_model.save_pretrained(final_model_dir)
    logger.info(f"Training completed. Final model saved to {final_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument("--model_path", type=str, default="/mnt/e/CheckPoints/ChemicalFactory/output/qwen205_moltrans_mit_mixed_augm_rlhf_sft_lora_para1",
                        help="Path to the pre-trained model")
    parser.add_argument("--dpo_data_path", type=str, default="/mnt/e/DataSets/Chemistry/ForwardPrediction/DPO",
                        help="Path to the PPO data directory")
    parser.add_argument("--dpo_data_file", type=str, default="MIT_mixed_augm.json",
                        help="Name of the PPO data file")
    parser.add_argument("--output_path", type=str, default="/mnt/e/CheckPoints/ChemicalFactory/output/qwen205_moltrans_mit_mixed_augm_dpo_lora_para2",
                        help="Path to save the trained model and checkpoints")
    parser.add_argument("--log_path", type=str, default="/mnt/e/CheckPoints/ChemicalFactory/output/qwen205_moltrans_mit_mixed_augm_dpo_lora_para2/training.log",
                        help="Path to the log file")
    parser.add_argument("--num_samples", type=int, default=10000000000000,
                        help="Number of samples to use for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--checkpoint_steps", type=int, default=500, help="Checkpoint saving frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_dpo(
        model_path=args.model_path,
        dpo_data_path=args.dpo_data_path,
        dpo_data_file=args.dpo_data_file,
        output_path=args.output_path,
        log_path=args.log_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed
    )