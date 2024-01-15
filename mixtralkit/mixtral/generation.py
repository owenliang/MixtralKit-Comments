# Copyright (c) OpenMMLab. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from mixtralkit.layers import (
    Tokenizer,
    MoETorchTransformer,
    MixtralModelArgs
)
from mixtralkit.utils import sample_top_p
from mixtralkit.utils.generation import (
    CompletionPrediction,
    Dialog,
    ChatPrediction

)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Mixtral:
    
    # 工厂方法，创建model对象
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        num_gpus: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Mixtral":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        model_parallel_size = 1

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()
        
        # 加载torch模型参数
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[0] #其实只加载了第1个torch模型文件
        
        # 模型结构的参数配置，和checkpoint配套的
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # 模型结构的描述参数，和checkpoint配套的
        model_args = MixtralModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_gpus=num_gpus,
            **params,
        )
        
        # 分词器，从磁盘加载词典
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        
        # torch默认采用float16
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        
        ######### 加载transfomer模型 #########
        model = MoETorchTransformer(model_args)
        print(f"=== created Mixtral 8x7B. Experts spread over {num_gpus} GPUs ===")
        
        # 取该Model所有的模型参数，由于每个参数有唯一的path名，可以配合torch还原checkpoints到各个参数，这就是模型复现的基本逻辑
        model_param_keys = []
        for key, value in model.named_parameters():
            model_param_keys.append(key)
        
        # 从磁盘上加载checkpoint文件，里面就是param name -> param value
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print("Total number of model parameters:{}".format(len(model_param_keys)))
        print("Total number of checkpoint parameters:{}".format(len(checkpoint)))

        # 模型还原param权重
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        # 返回 分词器+模型，向上层返回一个简单的chat对话接口
        return Mixtral(model, tokenizer)
    
    def __init__(self, model: MoETorchTransformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # 这是推理的基础方法，直面transfomer模型的输入和输出过程
    @torch.inference_mode() # 推理模式，不需要为反向传播记录中间计算结果，节省显存
    def generate(
        self,
        prompt_tokens: List[List[int]], # 输入的token列表，支持batch
        max_gen_len: int,   # 推理生成的最大长度，如果一直没有EOS出现则强制结束继续推理
        temperature: float = 0.6,   # softmax处理，控制多样性
        top_p: float = 0.9,     # probs概率处理，控制多样性
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)    # batch大小
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens) # batch中样本的最少token数
        max_prompt_len = max(len(t) for t in prompt_tokens) # batch中样本的最多token数
        assert max_prompt_len <= params.max_seq_len # 输入token数不能超过transformer最大能接受的输入长度
        
        # batch中每个样本要补齐sequence长度,短的补pad token
        # TODO：max_prompt_len+max_gen_len没理解，我理解max_prompt_len就好了
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # 生成transformer模型输入，（batch_size,total_len)先填充满padding，对齐batch输入的各个seq，以便批量计算
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        
        # 为每个样本分别填充实际输入token部分
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            
        # TODO: 生成和输入batch（batch_size,total_len)一样形状的全0矩阵，暂时看不出这是要做什么
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        
        # batch中每个样本对应一个bool值，标识模型输出有没有遇到过EOS
        eos_reached = torch.tensor([False] * bsz, device="cuda")    
        
        # padding掩码，注意力矩阵中无效的Q&K打分做清0用
        input_text_mask = tokens != pad_id
        
        # 特殊情况先不考虑
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        
        # TODO: 这里非常高级，用了kv cache方案优化注意力打分计算，暂时先不看细节
        # 大概意思就是输入token是可以分批进入model，不必一次性进入做注意力
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)  # 送入一段token序列
            if temperature > 0: # 多样性方案，将decoder输出logis/temp再做softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p) # 同时top_p累计范围内的probs都有几率被选中
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)    # 非多样性方案，就是每条样本输出序列中最后一个token的probs分布，选最大概率作为下一个token id

            next_token = next_token.reshape(-1)
            
            # TODO: 这块很高级, 似乎在前期是将真正的input tokens片段拿去推理，返回的Next token又拼接回input tokens末尾（不会覆盖真实input token，只会在原本的input tokens后面开始追加）
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            
            # TODO: 这块很高级，输出竟然直接拼回了输入，继续下一轮投入模型, 似乎想在model里复用上一次推理的cache, 肯定是kv cache那一套逻辑需要后面详细看
            tokens[:, cur_pos] = next_token # 记录下一个token到各个样本，可能是原始input还没结束，可能是已经推理出下一个token了
            # 这块先不用看了
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            
            # 当前input序列的末尾token是padding且预测出的是eos，那么更新各个样本的推理结束标识
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            
            # 全部序列都出现过EOS，结束推理
            if all(eos_reached):
                break
                
        # 不重要
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        
        # 批次推理结束，整理每个样本的推理结果
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):  # 遍历每个样本的输入+输出合并序列
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])    # 跳过toks中的输入token部分
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]    # 直至末尾
            
            # 不重要
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            
            # 截取至序列末尾的EOS
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]   # 截取到eos
                probs = probs[:eos_idx] if logprobs else None
            
            # 记录该样本的推理结果
            out_tokens.append(toks)
            # 不重要
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]
