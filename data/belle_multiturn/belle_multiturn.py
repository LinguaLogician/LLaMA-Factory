# Copyright 2025 the LlamaFactory team.
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import json
import os

import datasets


# _HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

_DESCRIPTION = "BELLE multiturn chat dataset."

_CITATION = """\
@article{belle2023exploring,
  title={Exploring the Impact of Instruction Data Scaling on Large Language Models},
  author={Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, Xiangang Li},
  journal={arXiv preprint arXiv:2303.14742},
  year={2023}
}
"""

_HOMEPAGE = f"{_HF_ENDPOINT}/datasets/BelleGroup/multiturn_chat_0.8M"
_LICENSE = "gpl-3.0"
_URL = f"{_HF_ENDPOINT}/datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/multiturn_chat_0.8M.json"


class BelleMultiturn(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}]}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download(_URL)
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                conversations = []
                prompt = data["instruction"].strip()
                response = data["output"].strip()

                assist_idx = prompt.rfind("Assistant:")
                human_idx = prompt.rfind("Human:")
                query = prompt[human_idx + 6 : assist_idx].strip()
                prompt = prompt[:human_idx].strip()
                conversations.insert(0, {"from": "gpt", "value": response})
                conversations.insert(0, {"from": "human", "value": query})

                while prompt.rfind("Assistant:") != -1:
                    assist_idx = prompt.rfind("Assistant:")
                    human_idx = prompt.rfind("Human:")
                    if human_idx != -1:
                        old_query = prompt[human_idx + 6 : assist_idx].strip()
                        old_resp = prompt[assist_idx + 10 :].strip()
                        conversations.insert(0, {"from": "gpt", "value": old_resp})
                        conversations.insert(0, {"from": "human", "value": old_query})
                    else:
                        break
                    prompt = prompt[:human_idx].strip()

                yield key, {"conversations": conversations}
