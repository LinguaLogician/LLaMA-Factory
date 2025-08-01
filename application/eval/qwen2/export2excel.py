import os
import re
from datetime import datetime
import pandas as pd
'''
请为我写一段Python代码，为我处理txt文件，将信息导入到Excel表格中
已知：
1. scores_dir=/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/scores
2. scores_dir之下有若干个文件夹，每个文件夹下有若干个txt文件，记录着大模型最后的评分信息，相关内容如下为例：

```
PREDICTION_FILE: /home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction/mit_mixed_augm_nospace_test/qwen205_moltrans_mit_mixed_augm_nospace_lora_para1_ckptlast.json
Total samples: 40000
Beam size: 5

Top-k Accuracy:
Top-1: 0.7950
Top-2: 0.8565
Top-3: 0.8762
Top-5: 0.8908

K-th Accuracy:
K=1: 0.7950
K=2: 0.1727
K=3: 0.1024
K=5: 0.0495

Top-k Validity Rate:
Top-1: 0.9924
Top-2: 0.9718
Top-3: 0.9532
Top-5: 0.9218

K-th Validity Rate:
K=1: 0.9924
K=2: 0.9512
K=3: 0.9159
K=5: 0.8620
```
3. excels_dir=/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/excels
要求：
1. 遍历每个文件夹中的每个文件，将txt中的信息做整理，利用表格工具生成相关excel文件，导出到excels_dir下，文件名为evaluation_scores_{时间戳}
2. 要将PREDICTION_FILE文件信息中的test set和model, checkpoint信息提取出来，在上述示例中，即为mit_mixed_augm_nospace_test, qwen205_moltrans_mit_mixed_augm_nospace_lora_para1和ckptlast。其他的同样如此
3. 制作单元格过程中，要注意同类相合并
	纵向：
	1. test set相同的行连续放置，且test set合并为一个大的单元格
	2. 在一个test set下，model相同的行连续放置，且model合并为一个大的单元格
	横向：
	1. Top-k Accuracy、 K-th Accuracy、Top-k Validity Rate和K-th Validity Rate大单元格下分为其响应的若干个子列
		Top-k Accuracy： Top-1, Top-2, Top-3, Top-5
		K-th Accuracy: K=1, K=2, K=3, K=5
		Top-k Validity Rate: Top-1, Top-2, Top-3, Top-5
		K-th Validity Rate: K=1, K=2, K=3, K=5
4. 相关过程封装成方法，但不要过度封装，从__main__中开始执行程序
5. 程序中不要出现任何中文
'''
def extract_info_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        content = f.read()

    pattern = r'PREDICTION_FILE: (.+?)\nTotal samples: (\d+)\nBeam size: (\d+)\n\n' \
              r'Top-k Accuracy:\nTop-1: ([\d.]+)\nTop-2: ([\d.]+)\nTop-3: ([\d.]+)\nTop-5: ([\d.]+)\n\n' \
              r'K-th Accuracy:\nK=1: ([\d.]+)\nK=2: ([\d.]+)\nK=3: ([\d.]+)\nK=5: ([\d.]+)\n\n' \
              r'Top-k Validity Rate:\nTop-1: ([\d.]+)\nTop-2: ([\d.]+)\nTop-3: ([\d.]+)\nTop-5: ([\d.]+)\n\n' \
              r'K-th Validity Rate:\nK=1: ([\d.]+)\nK=2: ([\d.]+)\nK=3: ([\d.]+)\nK=5: ([\d.]+)'

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    prediction_file = match.group(1)
    total_samples = int(match.group(2))
    beam_size = int(match.group(3))

    test_set = os.path.basename(os.path.dirname(prediction_file))
    model_ckpt = os.path.splitext(os.path.basename(prediction_file))[0]
    parts = model_ckpt.split("_")
    model = "_".join(parts[:-1])
    checkpoint = parts[-1]

    return {
        "Test_Set": test_set,
        "Model": model,
        "Checkpoint": checkpoint,
        "Top-1 Acc": float(match.group(4)),
        "Top-2 Acc": float(match.group(5)),
        "Top-3 Acc": float(match.group(6)),
        "Top-5 Acc": float(match.group(7)),
        "K=1 Acc": float(match.group(8)),
        "K=2 Acc": float(match.group(9)),
        "K=3 Acc": float(match.group(10)),
        "K=5 Acc": float(match.group(11)),
        "Top-1 Valid": float(match.group(12)),
        "Top-2 Valid": float(match.group(13)),
        "Top-3 Valid": float(match.group(14)),
        "Top-5 Valid": float(match.group(15)),
        "K=1 Valid": float(match.group(16)),
        "K=2 Valid": float(match.group(17)),
        "K=3 Valid": float(match.group(18)),
        "K=5 Valid": float(match.group(19)),
        "Total_Samples": total_samples,
        "Beam_Size": beam_size
    }

def checkpoint_sort_key(ckpt):
    if ckpt == "ckptlast":
        return float('inf')
    match = re.match(r"ckpt(\d+)k", ckpt)
    return int(match.group(1)) if match else -1

def process_scores_dir(scores_dir, excels_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"evaluation_scores_{timestamp}.xlsx"
    excel_path = os.path.join(excels_dir, excel_filename)

    all_rows = []

    for dir in sorted(os.listdir(scores_dir)):
        dir_path = os.path.join(scores_dir, dir)
        if not os.path.isdir(dir_path):
            continue

        for file in sorted(os.listdir(dir_path)):
            if not file.endswith(".txt"):
                continue
            txt_file_path = os.path.join(dir_path, file)
            info = extract_info_from_txt(txt_file_path)
            if info:
                all_rows.append(info)

    if not all_rows:
        print("No valid score data found.")
        return

    df = pd.DataFrame(all_rows)

    # Apply sorting
    df["Checkpoint_Sort_Key"] = df["Checkpoint"].map(checkpoint_sort_key)
    df = df.sort_values(by=["Test_Set", "Model", "Checkpoint_Sort_Key"]).drop(columns="Checkpoint_Sort_Key")

    # Write to Excel with merged cells and auto column width
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Evaluation", index=False, startrow=1, header=False)
        workbook = writer.book
        worksheet = writer.sheets["Evaluation"]

        # Write headers with formatting
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Auto column widths
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(str(col)) + 2)
            worksheet.set_column(i, i, max_len)

        # Merge same Test_Set and Model vertically
        def merge_cells(col_idx, col_name, group_cols):
            grouped = df.groupby(group_cols, sort=False)
            start_row = 1
            for _, group in grouped:
                same_val = group[col_name].iloc[0]
                row_count = len(group)
                if row_count > 1:
                    worksheet.merge_range(start_row, col_idx, start_row + row_count - 1, col_idx, same_val)
                else:
                    worksheet.write(start_row, col_idx, same_val)
                start_row += row_count

        merge_cells(0, "Test_Set", ["Test_Set"])
        merge_cells(1, "Model", ["Test_Set", "Model"])

if __name__ == "__main__":
    scores_dir = "/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/scores3"
    excels_dir = "/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/excels"
    os.makedirs(excels_dir, exist_ok=True)
    process_scores_dir(scores_dir, excels_dir)
