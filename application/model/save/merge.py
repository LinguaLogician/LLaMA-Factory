# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: merge2.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/4 15:00
'''
请为我编写Python程序，
已知：
1. 有一个Python程序的执行命令是这样的：
llamafactory-cli export ${yaml_relative_path}
yaml_relative_path可能为examples/merge_lora/qwen205/space/qwen205_moltrans_mit_separated_space_lora_para1.yaml
它是相对项目根目录project_path所在路径
2. yaml文件的内容如下：
```
adapter_name_or_path: ${adapter_path}/${checkpoint}
export_dir: ${output_dir}/${ckpt}
```
3. adapter_name_or_path为lora训练后的增量部分checkpoint的地址, adapter_path、checkpoint变量由外部传入
4. export_dir指向的是模型最后导出的目录, output_dir、ckpt变量由外部传入
5. checkpoint格式为f"checkpoint-{steps}", 例如：'checkpoint-10000'，在一个adapter_path下，有多个checkpoint目录，steps从小到达排开，其中，最后一个checkpoint目录为模型最后训练得到的checkpoint，steps与其他整位数不同，如：'checkpoint-300639'
6. ckpt为checkpoint 名字的简写，如'checkpoint-10000'的ckpt为'ckpt10k',最后一个checkpoint可以简写为 ckptlast
7. 项目绝对路径project_path=/home/liangtao/Development/LLMSpace/LLaMA-Factory/
要求
1. 根据要求，在python中执行上述llamafactory-cli export 命令, 要从python程序传入参数变量到该命令进行执行，并可覆盖yaml文件中的参数
2. 由于Python程序存放的位置可能在project_path子目录中，所以要确保执行llamafactory-cli export ${yaml_relative_path}命令时，不会导致路径出错
3. 设置变量表示merge间隔，变量名由你定义，例如设置merge间隔为20000，就意味着每checkpoint变量的steps可能为20000、40000、60000，依次类推
4. 设置变量表示是否保存最后一个checkpoint
5. 设置变量表示额外定义的checkpoint可能不在第3、4条要求中，例如merge间隔为20000, 但我要求保存steps为30000的checkpoint, 此变量为一数组，若为空，表示没有额外需要merge的checkpoint
6. 对于第3项要求，设置变量是否做规律性保存，如果是，则按照第三项merge保存，如果否则检查可否按第5条要求merge保存，如果第5条也没有，则值merge保存最后一个checkpoint
7. 对于第3项，设置变量start_ckpt, 如:start_ckpt=80000, 即表示，从checkpoint为"checkpoint-80000"开始按照第3条执行，忽略前面项。若此参数没有设置，则默认从最小merge间隔数开始，例如，若merge间隔为20000，则从20000开始
8. 程序通过arg parser设置变量，并可以通过外部shell执行脚本传入变量，不要在方法内设置配置变量
9. 处理过程中，显示进度条
10. 方法进行适度封装，不要封装过度，从__main__开始执行，arg parser设置和处理变量部分，直接放在__main__部分
11. 设置变量，表示merge处理线程的个数，用一个线程池维护，merge过程中，可以并发执行处理
12. 代码中不要有中文日志或注释
'''
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil


def find_checkpoints(adapter_path, merge_interval, save_last, extra_checkpoints, regular_save, start_ckpt=None):
    """
    Find all checkpoints to process based on the given parameters.
    """
    checkpoints = []

    # Get all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(adapter_path) if d.startswith('checkpoint-')]
    if not checkpoint_dirs:
        return checkpoints

    # Extract steps and sort
    steps = []
    for d in checkpoint_dirs:
        try:
            step = int(d.split('-')[1])
            steps.append(step)
        except (IndexError, ValueError):
            continue

    if not steps:
        return checkpoints

    steps.sort()
    max_step = max(steps)

    # Handle regular save
    if regular_save:
        current = start_ckpt if start_ckpt is not None else merge_interval
        while current <= max_step:
            if current in steps:
                checkpoints.append(f"checkpoint-{current}")
            current += merge_interval

    # Handle extra checkpoints
    for step in extra_checkpoints:
        if step in steps and f"checkpoint-{step}" not in checkpoints:
            checkpoints.append(f"checkpoint-{step}")

    # Handle last checkpoint
    if save_last and max_step not in [int(c.split('-')[1]) for c in checkpoints]:
        checkpoints.append(f"checkpoint-{max_step}")

    # If no checkpoints found and not regular save and no extra checkpoints, use last
    if not checkpoints and not regular_save and not extra_checkpoints and save_last:
        checkpoints.append(f"checkpoint-{max_step}")

    return sorted(checkpoints, key=lambda x: int(x.split('-')[1])), max_step


def create_temp_yaml(original_yaml_path, adapter_path, checkpoint, output_dir, ckpt_name):
    """
    Create a temporary YAML file with the parameters substituted.
    """
    # Read original YAML
    with open(original_yaml_path, 'r') as f:
        content = f.read()

    # Substitute variables
    content = content.replace('${adapter_path}', adapter_path)
    content = content.replace('${checkpoint}', checkpoint)
    content = content.replace('${output_dir}', output_dir)
    content = content.replace('${ckpt}', ckpt_name)

    # Create temp file
    temp_dir = tempfile.mkdtemp()
    temp_yaml_path = os.path.join(temp_dir, os.path.basename(original_yaml_path))

    with open(temp_yaml_path, 'w') as f:
        f.write(content)

    return temp_yaml_path, temp_dir


def process_checkpoint(yaml_relative_path, project_path, adapter_path, output_dir, checkpoint, ckpt_name):
    """
    Process a single checkpoint by executing the export command.
    """
    # Ensure we're in the project root directory
    original_dir = os.getcwd()
    os.chdir(project_path)

    try:
        # Create temporary YAML with substituted values
        original_yaml_path = os.path.join(project_path, yaml_relative_path)
        temp_yaml_path, temp_dir = create_temp_yaml(
            original_yaml_path=original_yaml_path,
            adapter_path=adapter_path,
            checkpoint=checkpoint,
            output_dir=output_dir,
            ckpt_name=ckpt_name
        )

        # Prepare command (without additional parameters)
        cmd = f"llamafactory-cli export {temp_yaml_path}"

        # Execute command
        exit_code = os.system(cmd)

        # Clean up temp file
        shutil.rmtree(temp_dir)

        return exit_code == 0
    except Exception as e:
        print(f"Error processing {checkpoint}: {str(e)}")
        return False
    finally:
        os.chdir(original_dir)


def get_ckpt_name(checkpoint, is_last=False):
    """
    Generate the ckpt name based on checkpoint steps.
    """
    if is_last:
        return "ckptlast"

    try:
        step = int(checkpoint.split('-')[1])
        if step >= 1000:
            return f"ckpt{step // 1000}k"
        return f"ckpt{step}"
    except (IndexError, ValueError):
        return "ckptunknown"


def main(args):
    # Find all checkpoints to process
    checkpoints, max_step = find_checkpoints(
        adapter_path= os.path.join(args.project_path, args.adapter_path),
        merge_interval=args.merge_interval,
        save_last=args.save_last,
        extra_checkpoints=args.extra_checkpoints,
        regular_save=args.regular_save,
        start_ckpt=args.start_ckpt
    )

    if not checkpoints:
        print("No checkpoints found to process.")
        return

    print(f"Found {len(checkpoints)} checkpoints to process:")
    for c in checkpoints:
        print(f"- {c}")

    # Process checkpoints with thread pool
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        # max_step = max(int(c.split('-')[1]) for c in checkpoints)

        for checkpoint in checkpoints:
            is_last = (int(checkpoint.split('-')[1]) == max_step)
            ckpt_name = get_ckpt_name(checkpoint, is_last)

            futures.append(executor.submit(
                process_checkpoint,
                yaml_relative_path=args.yaml_relative_path,
                project_path=args.project_path,
                adapter_path=args.adapter_path,
                output_dir=args.output_dir,
                checkpoint=checkpoint,
                ckpt_name=ckpt_name
            ))

        # Show progress
        with tqdm(total=len(futures), desc="Processing checkpoints") as pbar:
            for future in as_completed(futures):
                future.result()  # Raise exception if any
                pbar.update(1)

    print("All checkpoints processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LLaMA-Factory checkpoints for export.")

    # Required arguments
    parser.add_argument("--yaml_relative_path", type=str,
                        default="examples/merge_lora/qwen205/common.yaml",
                        help="Relative path to YAML config from project root")
    parser.add_argument("--project_path", type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory",
                        help="Absolute path to project root directory")
    parser.add_argument("--adapter_path", type=str,
                        default="saves/sft_lora/qwen205_moltrans_mit_mixed_space_lora_para1",
                        help="Path to adapter checkpoints")
    parser.add_argument("--output_dir", type=str,
                        default="output/qwen205_moltrans_mit_mixed_space_lora_para1",
                        help="Output directory for exported models")

    # Processing parameters
    parser.add_argument("--merge_interval", type=int, default=20000,
                        help="Interval between checkpoints to process")
    parser.add_argument("--save_last",
                        default=True,
                        # action="store_true",
                        help="Whether to save the last checkpoint")
    parser.add_argument("--extra_checkpoints", type=int, nargs="*",
                        default=[],
                        help="Extra checkpoint steps to process")
    parser.add_argument("--regular_save",
                        # action="store_true",
                        default=True,
                        help="Whether to save checkpoints at regular intervals")
    parser.add_argument("--start_ckpt", type=int, default=80000,
                        help="Starting checkpoint step for regular processing")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads to use for parallel processing")

    args = parser.parse_args()

    # Convert project_path to absolute path and normalize
    args.project_path = str(Path(args.project_path).absolute())

    main(args)


# python script.py \
#     --yaml_relative_path examples/merge_lora/qwen205/space/qwen205_moltrans_mit_separated_space_lora_para1.yaml \
#     --project_path /home/liangtao/Development/LLMSpace/LLaMA-Factory/ \
#     --adapter_path /path/to/adapters \
#     --output_dir /path/to/output \
#     --merge_interval 20000 \
#     --save_last \
#     --extra_checkpoints 30000 50000 \
#     --regular_save \
#     --threads 4