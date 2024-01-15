# Copyright (c) OpenMMLab. and affiliates.

import argparse
from mixtralkit.mixtral import Mixtral


def parse_args():
    parser = argparse.ArgumentParser(description='Run an inference of mixtral-8x7b model')
    parser.add_argument('-m',
                        '--model-weights',
                        help='Model weights.',
                        default=None,
                        type=str)
    parser.add_argument('-t',
                        '--tokenizer',
                        help='path of tokenizer file.',
                        default=None,
                        type=str)
    parser.add_argument('--num-gpus', type=int)

    args = parser.parse_args()
    return args


def main():
    # 命令行参数
    args = parse_args()
    
    # transformer模型限制
    max_batch_size = 4  
    max_seq_len = 1024  # 输入序列最大token数
    max_gen_len = 64
    
    # 多个测试用例
    prompts = [
        "Who are you?",
        "1 + 1 -> 3\n"
        "2 + 2 -> 5\n"
        "3 + 3 -> 7\n"
        "4 + 4 -> ",
        "请问你是什么模型？",
        ]

    # 推理多样性控制
    temperature = 1.0 # for greedy decoding
    top_p = 0.9

    # 构建Mixtral模型
    generator = Mixtral.build(
        ckpt_dir=args.model_weights,    # 模型参数
        tokenizer_path=args.tokenizer,  # 分词器
        max_seq_len=max_seq_len,    # 输入序列最大长度
        max_batch_size=max_batch_size, # batch最大限制
        num_gpus=args.num_gpus, # GPU数量
    )
    
    # 批量推理多个测试用例
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,    # 限制transformer输出序列长度
        # 多样性控制
        temperature=temperature,  
        top_p=top_p,
    )
    
    # 打印每个用例的输入和输出
    for prompt, result in zip(prompts, results):
        print("="*30 + "Example START" + "="*30 + '\n')
        print("[Prompt]:\n{}\n".format(prompt))
        print("[Response]:\n{}\n".format(result['generation']))
        print("="*30 + "Example END" + "="*30 + '\n')

if __name__ == "__main__":
    main()