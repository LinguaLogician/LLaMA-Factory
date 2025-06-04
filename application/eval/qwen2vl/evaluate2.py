import argparse
import json
import os.path

from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import shortuuid

from transformers import AutoProcessor
from vllm import SamplingParams, LLM
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 假设图像路径需要从样本中获取并加载为Tensor
        image = Image.open(sample['image']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return sample['question_id'], image, sample['text'], sample['category'], sample['topic']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    MODEL_PATH = "/data/liangtao/Models/Qwen/Qwen2-VL-7B-Instruct"

    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小以适应模型输入
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    answers_dir = os.path.expanduser(os.path.join(args.eval_dir, "answers", args.model_name))
    images_dir = os.path.expanduser(os.path.join(args.eval_dir, "images"))
    questions_dir = os.path.expanduser(os.path.join(args.eval_dir, "questions"))
    os.makedirs(answers_dir, exist_ok=True)
    for question_file in os.listdir(questions_dir):
        if not question_file.endswith(".jsonl"):
            continue
        # 创建数据集实例
        dataset = CustomDataset(os.path.join(questions_dir, question_file), transform=transform)
        # 使用 DataLoader
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

        category = os.path.basename(question_file).split('_')[0]
        with open(os.path.join(answers_dir, f"{category}_answers.jsonl"), "w") as ans_file:
            # data_loader = create_data_loader(questions, images_dir, text_processor, image_processor)
            # model.to(device='cuda')
            for _, (question_ids, images, questions, categories, topics) in tqdm(enumerate(data_loader), total=len(data_loader), desc=question_file):

                sampling_params = SamplingParams(
                    temperature=0.1,
                    top_p=0.001,
                    repetition_penalty=1.05,
                    max_tokens=256,
                    stop_token_ids=[],
                )

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                                "min_pixels": 50 * 50,
                                "max_pixels": 1280 * 28 * 28,
                            },
                            {"type": "text", "text": "What is the text in the illustrate?"},
                        ],
                    },
                ]
                processor = AutoProcessor.from_pretrained(MODEL_PATH)
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)

                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                generated_text = outputs[0].outputs[0].text

                for question_id, question, annotation, topic in zip(question_ids, questions, annotations,
                                                                            topics):
                    ans_file.write(json.dumps({"question_id": question_id,
                                               "prompt": question,
                                               "text": generated_text,
                                               "answer_id": shortuuid.uuid(),
                                               "category": category,
                                               "topic": topic,
                                               "metadata": {}}, ensure_ascii=False) + "\n")
