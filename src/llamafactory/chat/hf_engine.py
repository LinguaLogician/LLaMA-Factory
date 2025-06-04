# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from collections.abc import AsyncGenerator
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Tuple, List

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from typing_extensions import override
import torch.nn.functional as F

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, EngineName
from ..model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class HuggingfaceEngine(BaseEngine):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.name = EngineName.HF
        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left" if self.can_generate else "right"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.model = load_model(
            self.tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=(not self.can_generate)
        )  # must after fixing tokenizer to resize vocab
        self.generating_args = generating_args.to_dict()
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            logger.warning_rank0_once("There is no current event loop, creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))

    @staticmethod
    def _process_args(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: dict[str, Any],
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> tuple[dict[str, Any], int]:
        mm_input_dict = {"images": [], "videos": [], "audios": [], "imglens": [0], "vidlens": [0], "audlens": [0]}
        if images is not None:
            mm_input_dict.update({"images": images, "imglens": [len(images)]})
            if not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

        if videos is not None:
            mm_input_dict.update({"videos": videos, "vidlens": [len(videos)]})
            if not any(VIDEO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = VIDEO_PLACEHOLDER * len(videos) + messages[0]["content"]

        if audios is not None:
            mm_input_dict.update({"audios": audios, "audlens": [len(audios)]})
            if not any(AUDIO_PLACEHOLDER in message["content"] for message in messages):
                messages[0]["content"] = AUDIO_PLACEHOLDER * len(audios) + messages[0]["content"]

        messages = template.mm_plugin.process_messages(
            messages, mm_input_dict["images"], mm_input_dict["videos"], mm_input_dict["audios"], processor
        )
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        system = system or generating_args["default_system"]
        enable_thinking = input_kwargs.pop("enable_thinking", None)
        enable_thinking = enable_thinking if enable_thinking is not None else generating_args["enable_thinking"]
        prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, system, tools, enable_thinking)
        prompt_ids, _ = template.mm_plugin.process_token_ids(
            prompt_ids,
            None,
            mm_input_dict["images"],
            mm_input_dict["videos"],
            mm_input_dict["audios"],
            tokenizer,
            processor,
        )
        prompt_length = len(prompt_ids)
        inputs = torch.tensor([prompt_ids], device=model.device)
        attention_mask = torch.ones_like(inputs, dtype=torch.long)

        do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        return_dict_in_generate: bool = input_kwargs.pop("return_dict_in_generate", False)
        output_scores: bool = input_kwargs.pop("output_scores", False)
        repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
        skip_special_tokens: Optional[bool] = input_kwargs.pop("skip_special_tokens", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
        stop: Optional[Union[str, list[str]]] = input_kwargs.pop("stop", None)

        if stop is not None:
            logger.warning_rank0("Stop parameter is not supported by the huggingface engine yet.")

        generating_args = generating_args.copy()
        generating_args.update(
            dict(
                do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
                temperature=temperature if temperature is not None else generating_args["temperature"],
                top_p=top_p if top_p is not None else generating_args["top_p"],
                top_k=top_k if top_k is not None else generating_args["top_k"],
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
                repetition_penalty=repetition_penalty
                if repetition_penalty is not None
                else generating_args["repetition_penalty"],
                length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
                skip_special_tokens=skip_special_tokens
                if skip_special_tokens is not None
                else generating_args["skip_special_tokens"],
                eos_token_id=template.get_stop_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        # if isinstance(num_return_sequences, int) and num_return_sequences > 1:  # do_sample needs temperature > 0
        #     generating_args["do_sample"] = True
        #     generating_args["temperature"] = generating_args["temperature"] or 1.0

        if not generating_args["temperature"]:
            generating_args["do_sample"] = False

        if not generating_args["do_sample"]:
            generating_args.pop("temperature", None)
            generating_args.pop("top_p", None)

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(**generating_args),
        )

        mm_inputs = template.mm_plugin.get_mm_inputs(**mm_input_dict, batch_ids=[prompt_ids], processor=processor)
        for key, value in mm_inputs.items():
            if isinstance(value, list) and isinstance(value[0], torch.Tensor):  # for pixtral inputs
                value = torch.stack(value)  # assume they have same sizes
            elif (
                isinstance(value, list) and isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor)
            ):  # for minicpmv inputs
                value = torch.stack([torch.stack(v) for v in value])
            elif not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            if torch.is_floating_point(value):  # cast data dtype for paligemma
                value = value.to(model.dtype)

            if key == "second_per_grid_ts":  # qwen2.5vl special case
                gen_kwargs[key] = value.tolist()
            else:
                gen_kwargs[key] = value.to(model.device)

        if getattr(model.config, "model_type", None) in ["minicpmv", "minicpmo"]:
            gen_kwargs["input_ids"] = inputs
            gen_kwargs["tokenizer"] = tokenizer
            if "audio_feature_lens" in mm_inputs:
                gen_kwargs["audio_feature_lens"] = mm_inputs["audio_feature_lens"]

            gen_kwargs.pop("image_sizes", None)

        return gen_kwargs, prompt_length


    @staticmethod
    def _process_llm_args(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        template: "Template",
        generating_args: dict[str, Any],
        messages_batch: list[list[dict[str, str]]],
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> tuple[dict[str, Any], list[int]]:

        prompt_ids_list = []
        prompt_length_list = []
        for messages in messages_batch:
            paired_messages = messages + [{"role": "assistant", "content": ""}]
            prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages)
            prompt_length_list.append(len(prompt_ids))
            prompt_ids_list.append(prompt_ids)
        inputs = tokenizer.pad(
            {"input_ids": prompt_ids_list},
            padding="longest",
            return_tensors="pt"
        ).to(model.device)

        inputs=inputs["input_ids"]
        # inputs = torch.tensor(prompt_ids_list, device=model.device)
        attention_mask = torch.ones_like(inputs, dtype=torch.long)

        do_sample: Optional[bool] = input_kwargs.pop("do_sample", None)
        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        return_dict_in_generate: bool = input_kwargs.pop("return_dict_in_generate", False)
        output_scores: bool = input_kwargs.pop("output_scores", False)
        repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        length_penalty: Optional[float] = input_kwargs.pop("length_penalty", None)
        skip_special_tokens: Optional[bool] = input_kwargs.pop("skip_special_tokens", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
        stop: Optional[Union[str, list[str]]] = input_kwargs.pop("stop", None)

        if stop is not None:
            logger.warning_rank0("Stop parameter is not supported by the huggingface engine yet.")

        generating_args = generating_args.copy()
        generating_args.update(
            dict(
                do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
                temperature=temperature if temperature is not None else generating_args["temperature"],
                top_p=top_p if top_p is not None else generating_args["top_p"],
                top_k=top_k if top_k is not None else generating_args["top_k"],
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
                repetition_penalty=repetition_penalty
                if repetition_penalty is not None
                else generating_args["repetition_penalty"],
                length_penalty=length_penalty if length_penalty is not None else generating_args["length_penalty"],
                skip_special_tokens=skip_special_tokens
                if skip_special_tokens is not None
                else generating_args["skip_special_tokens"],
                eos_token_id=template.get_stop_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
        )

        if isinstance(num_return_sequences, int) and num_return_sequences > 1:  # do_sample needs temperature > 0
            generating_args["do_sample"] = True
            generating_args["temperature"] = generating_args["temperature"] or 1.0

        if not generating_args["temperature"]:
            generating_args["do_sample"] = False

        if not generating_args["do_sample"]:
            generating_args.pop("temperature", None)
            generating_args.pop("top_p", None)

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=inputs,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(**generating_args),
        )
        return gen_kwargs, prompt_length_list


    @staticmethod
    @torch.inference_mode()
    def _chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: dict[str, Any],
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> list["Response"]:
        import math

        def are_all_inf_at_end(step_logprobs):
            first_inf_index = None
            for i, val in enumerate(step_logprobs):
                if math.isinf(val) and val < 0:
                    first_inf_index = i
                    break
            if first_inf_index is None:
                return True
            for val in step_logprobs[first_inf_index:]:
                if not (math.isinf(val) and val < 0):
                    return False
            return True

        def compute_beam_sequence_probs(output, prompt_length, messages=[]):
            sequences = output.sequences  # [num_return_sequences, total_length]
            scores = output.scores  # List of [num_beams, vocab_size] tensors (length: max_new_tokens)
            beam_indices = output.beam_indices  # [num_return_sequences, max_new_tokens]
            results = []
            num_return_sequences, total_seq_len = sequences.shape
            for beam_id in range(num_return_sequences):
                generated_token_ids = sequences[beam_id][prompt_length:]
                beam_path = beam_indices[beam_id]

                logprob_sum = 0.0
                step_logprobs = []

                for step, (token_id, beam_idx_at_step) in enumerate(zip(generated_token_ids, beam_path)):
                    step_scores = scores[step]  # [num_beams, vocab_size]
                    step_log_probs = F.log_softmax(step_scores, dim=-1)
                    token_logprob = step_log_probs[beam_idx_at_step, token_id].item()
                    if token_logprob != float('-inf'):
                        logprob_sum += token_logprob
                    step_logprobs.append(token_logprob)
                    if tokenizer.eos_token_id==token_id:
                        break
                all_inf_at_tail = are_all_inf_at_end(step_logprobs)
                prob = torch.exp(torch.tensor(logprob_sum)).item()
                if not all_inf_at_tail:
                    logger.info(f"Warning: -inf error for {messages}, with step_logprobs: {step_logprobs}, "
                                f"with prob: {prob}")
                results.append({
                    "beam_id": beam_id,
                    "logprob_sum": logprob_sum,
                    "prob": prob,
                    "token_ids": sequences[beam_id].tolist(),
                    "step_logprobs": step_logprobs
                })

            return results

        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model,
            tokenizer,
            processor,
            template,
            generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        generate_output = model.generate(**gen_kwargs)

        beam_probs = []

        if hasattr(generate_output, 'sequences_scores'):
            sequence_probs = torch.exp(generate_output.sequences_scores).tolist()
        if hasattr(generate_output, 'scores') and gen_kwargs['generation_config'].num_beams > 1:
            beam_probs = compute_beam_sequence_probs(generate_output, prompt_length, messages)
        if isinstance(generate_output, tuple):
            generate_output = generate_output[1][0]  # post-process the minicpm_o output
        if isinstance(generate_output, dict):
            generate_output = generate_output['sequences']
        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=getattr(gen_kwargs["generation_config"], "skip_special_tokens", True),
            clean_up_tokenization_spaces=True,
        )
        results = []

        for i in range(len(response)):
            eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    sequence_score=beam_probs[i]['prob'] if beam_probs and i < len(beam_probs) else None,
                    prompt_length=prompt_length,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )

        return results


    @staticmethod
    @torch.inference_mode()
    def _batch_chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        template: "Template",
        generating_args: dict[str, Any],
        messages_batch: list[list[dict[str, str]]],
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> list[list["Response"]]:
        import math

        def are_all_inf_at_end(step_logprobs):
            first_inf_index = None
            for i, val in enumerate(step_logprobs):
                if math.isinf(val) and val < 0:
                    first_inf_index = i
                    break
            if first_inf_index is None:
                return True
            for val in step_logprobs[first_inf_index:]:
                if not (math.isinf(val) and val < 0):
                    return False
            return True

        def compute_beam_sequence_probs(output, prompt_length, num_return_sequences, messages_batch=[]):
            sequences = output.sequences  # [num_return_sequences*batch_size, total_length]
            scores = output.scores  # List of [num_return_sequences*batch_size, vocab_size] tensors (length: max_new_tokens)
            beam_indices = output.beam_indices  # [num_return_sequences*batch_size, max_new_tokens]
            results = []
            num_return_sequences_with_batch, total_seq_len = sequences.shape
            for beam_id in range(num_return_sequences_with_batch):
                generated_token_ids = sequences[beam_id][prompt_length:]
                beam_path = beam_indices[beam_id]

                logprob_sum = 0.0
                step_logprobs = []

                for step, (token_id, beam_idx_at_step) in enumerate(zip(generated_token_ids, beam_path)):
                    step_scores = scores[step]  # [num_beams, vocab_size]
                    step_log_probs = F.log_softmax(step_scores, dim=-1)
                    token_logprob = step_log_probs[beam_idx_at_step, token_id].item()
                    if token_logprob != float('-inf'):
                        logprob_sum += token_logprob
                    step_logprobs.append(token_logprob)
                    if tokenizer.eos_token_id==token_id:
                        break
                all_inf_at_tail = are_all_inf_at_end(step_logprobs)
                prob = torch.exp(torch.tensor(logprob_sum)).item()
                if not all_inf_at_tail:
                    messages_id = beam_id//num_return_sequences
                    logger.info(f"Warning: -inf error for {messages_batch[messages_id]}, with step_logprobs: {step_logprobs}, "
                                f"with prob: {prob}")
                results.append({
                    "beam_id": beam_id,
                    "logprob_sum": logprob_sum,
                    "prob": prob,
                    "token_ids": sequences[beam_id].tolist(),
                    "step_logprobs": step_logprobs
                })

            return results

        gen_kwargs, prompt_length_list = HuggingfaceEngine._process_llm_args(
            model,
            tokenizer,
            template,
            generating_args,
            messages_batch,
            input_kwargs,
        )
        generate_output = model.generate(**gen_kwargs)
        num_return_sequences = gen_kwargs['generation_config'].num_return_sequences
        beam_probs = []
        if hasattr(generate_output, 'sequences_scores'):
            sequence_probs = torch.exp(generate_output.sequences_scores).tolist()
        if hasattr(generate_output, 'scores'):
            beam_probs = compute_beam_sequence_probs(generate_output, max(prompt_length_list), num_return_sequences,
                                                     messages_batch)
        if isinstance(generate_output, tuple):
            generate_output = generate_output[1][0]  # post-process the minicpm_o output
        if isinstance(generate_output, dict):
            generate_output = generate_output['sequences']
        response_ids = generate_output[:, max(prompt_length_list):]
        response = tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=getattr(gen_kwargs["generation_config"], "skip_special_tokens", True),
            clean_up_tokenization_spaces=True,
        )

        response_batch = []
        for i in range(len(messages_batch)):
            response_beam = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                eos_index = (response_ids[idx] == tokenizer.eos_token_id).nonzero()
                response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[idx])
                response_beam.append(
                    Response(
                    response_text=response[idx],
                    response_length=response_length,
                    sequence_score=beam_probs[idx]['prob'] if beam_probs and idx < len(beam_probs) else None,
                    prompt_length=prompt_length_list[i],
                    finish_reason="stop" if len(eos_index) else "length",
                ))
            response_batch.append(response_beam)

        return response_batch


    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        template: "Template",
        generating_args: dict[str, Any],
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> Callable[[], str]:
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model,
            tokenizer,
            processor,
            template,
            generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=getattr(gen_kwargs["generation_config"], "skip_special_tokens", True),
        )
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        def stream():
            try:
                return streamer.__next__()
            except StopIteration:
                raise StopAsyncIteration()

        return stream

    @staticmethod
    @torch.inference_mode()
    def _get_scores(
        model: "PreTrainedModelWrapper",
        tokenizer: "PreTrainedTokenizer",
        batch_input: list[str],
        input_kwargs: Optional[dict[str, Any]] = {},
    ) -> list[float]:
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        device = getattr(model.pretrained_model, "device", "cuda")
        inputs: dict[str, torch.Tensor] = tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        values: torch.Tensor = model(**inputs, return_dict=True, use_cache=False)[-1]
        scores = values.gather(dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return scores

    @override
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        input_args = (
            self.model,
            self.tokenizer,
            self.processor,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        async with self.semaphore:
            return await asyncio.to_thread(self._chat, *input_args)

    @override
    async def chat_llm_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        **input_kwargs,
    ) -> list[list["Response"]]:

        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        input_args = (
            self.model,
            self.tokenizer,
            self.template,
            self.generating_args,
            messages_batch,
            input_kwargs,
        )

        async with self.semaphore:
            return await asyncio.to_thread(self._batch_chat, *input_args)

    @override
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")

        input_args = (
            self.model,
            self.tokenizer,
            self.processor,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            images,
            videos,
            audios,
            input_kwargs,
        )
        async with self.semaphore:
            stream = self._stream_chat(*input_args)
            while True:
                try:
                    yield await asyncio.to_thread(stream)
                except StopAsyncIteration:
                    break

    @override
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        input_args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self.semaphore:
            return await asyncio.to_thread(self._get_scores, *input_args)
