# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: processor_internvl.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/1 14:23
# https://chat.deepseek.com/a/chat/s/c515b979-6a54-4b50-b745-d6f18e3b2da4
# https://chat.deepseek.com/a/chat/s/2ca4efb8-a508-410e-bd2e-b2928b136da8
# https://chat.deepseek.com/a/chat/s/92473a53-fc53-47b5-8c01-7e09e58b45f8
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from typing import List, Optional, Dict, Any, Union
from transformers import ProcessorMixin, AutoTokenizer, AutoConfig, AutoProcessor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import load_image


class InternVL2ImageProcessor(BaseImageProcessor):
    def __init__(
            self,
            image_size: int = 448,
            patch_size: int = 14,
            downsample_ratio: float = 0.5,
            min_patches: int = 1,
            max_patches: int = 12,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        # 添加image_seq_length属性以兼容LLaMA-Factory
        self.image_seq_length = self.num_image_token

    def preprocess(
            self,
            images: Union["ImageObject", List["ImageObject"]],
            crop_to_patches: bool = False,
            max_patches: Optional[int] = None,
            min_patches: Optional[int] = None,
            return_tensors: Optional[str] = "pt",
            **kwargs
    ) -> BatchFeature:
        """
        预处理图像，支持裁剪为patches的模式
        """
        if not isinstance(images, list):
            images = [images]

        # 转换为PIL图像
        images = [load_image(img) for img in images]

        if crop_to_patches:
            return self._preprocess_with_patches(
                images,
                max_patches=max_patches or self.max_patches,
                min_patches=min_patches or self.min_patches,
                return_tensors=return_tensors
            )
        else:
            return self._preprocess_normal(images, return_tensors=return_tensors)

    def _preprocess_normal(self, images: List[Image.Image], return_tensors: str = "pt") -> BatchFeature:
        """标准图像预处理"""
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

        transform = Compose([
            Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            CenterCrop((self.image_size, self.image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        pixel_values = [transform(img) for img in images]

        if return_tensors == "pt":
            pixel_values = torch.stack(pixel_values)

        return BatchFeature(data={"pixel_values": pixel_values, "num_patches": [1] * len(images)})

    def _preprocess_with_patches(
            self,
            images: List[Image.Image],
            max_patches: int = 12,
            min_patches: int = 1,
            return_tensors: str = "pt"
    ) -> BatchFeature:
        """将图像裁剪为多个patches"""
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

        # 基础变换
        base_transform = Compose([
            Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            CenterCrop((self.image_size, self.image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_patches = []
        num_patches_list = []

        for img in images:
            # 随机决定裁剪多少个patches
            num_patches = torch.randint(min_patches, max_patches + 1, (1,)).item()
            num_patches_list.append(num_patches)

            # 对每个patch进行预处理
            for _ in range(num_patches):
                patch_tensor = base_transform(img)
                all_patches.append(patch_tensor)

        if return_tensors == "pt":
            all_patches = torch.stack(all_patches)

        return BatchFeature(data={
            "pixel_values": all_patches,
            "num_patches": num_patches_list
        })

    def __call__(self, images, return_tensors="pt", **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


class InternVL2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "InternVL2ImageProcessor"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 手动设置类引用，避免依赖Auto系统
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        # 添加LLaMA-Factory需要的属性
        self.image_seq_length = getattr(image_processor, 'image_seq_length',
                                        getattr(image_processor, 'num_image_token', 256))

        super(ProcessorMixin, self).__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        重写from_pretrained方法
        """
        # 确保trust_remote_code=True
        kwargs.setdefault('trust_remote_code', True)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 创建image_processor
        image_processor = InternVL2ImageProcessor()

        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(self, text=None, images=None, **kwargs):
        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)
        else:
            image_inputs = {}

        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
        else:
            text_inputs = {}

        return {**text_inputs, **image_inputs}

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_names = getattr(self.tokenizer, 'model_input_names', [])
        return tokenizer_names + ["pixel_values", "num_patches"]


# 创建专门的加载函数
def load_internvl2_processor(model_path, **kwargs):
    """
    加载InternVL2处理器的便捷函数
    """
    # 确保trust_remote_code=True
    kwargs.setdefault('trust_remote_code', True)

    try:
        # 首先尝试直接使用InternVL2Processor的from_pretrained
        processor = InternVL2Processor.from_pretrained(model_path, **kwargs)
        return processor
    except Exception as e:
        print(f"InternVL2Processor failed, falling back to manual creation: {e}")
        # 回退到手动创建
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = InternVL2ImageProcessor()
        return InternVL2Processor(image_processor=image_processor, tokenizer=tokenizer)


if __name__ == '__main__':
    # 明确设置trust_remote_code=True
    processor = load_internvl2_processor(
        '/mnt/d/ChemicalFactory/output/internvl21_chemicals_retrosyn_full_para01',
        trust_remote_code=True
    )
    print("Processor loaded successfully:")
    print(f"Tokenizer: {type(processor.tokenizer)}")
    print(f"Image Processor: {type(processor.image_processor)}")
    print(f"Image sequence length: {processor.image_seq_length}")