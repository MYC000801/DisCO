# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.torch_functional import build_loss_mask

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTMTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 max_length=1024,
                 truncation='error'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()


    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
            
        self.prompts = self.prompts.tolist()


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        chat = self.prompts[item]


        # string
        chat_str = tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False) + tokenizer.eos_token
        chat_str = chat_str.replace(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            "<|im_start|>system\nYou are an intelligent agent navigating a maze.\nAt each step, you receive an observation of four adjacent cells, described by their coordinates and whether they are 'path' or 'wall'.\nYou must choose exactly one adjacent cell that is a valid 'path' or 'exit' and move into it.\nAlways move efficiently toward the goal.\nOutput your next move as a single coordinate in the format (row, col).\nDo not explain or repeat the input — just return the next move.\n<|im_end|>"
        )

        # tokenize
        chat_ids_output = tokenizer(chat_str, return_tensors='pt', add_special_tokens=False)
        chat_ids = chat_ids_output['input_ids'][0]
        chat_attention_mask = chat_ids_output['attention_mask'][0]

        im_start_id  = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id    = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        attention_mask = build_loss_mask(chat_ids, im_start_id, im_end_id, assistant_id)


        def highlight_loss_tokens(input_ids, loss_mask, tokenizer, *, color="red"):
            """
            将 input_ids 解码为字符串，并把 loss_mask==1 的 token 用颜色突出显示。

            Parameters
            ----------
            input_ids : torch.Tensor | list[int]
                长度 N 的 token id 序列（1-D）。
            loss_mask : torch.Tensor | list[int] / list[bool]
                与 input_ids 等长，0/1 或 False/True。
            tokenizer : transformers.PreTrainedTokenizer
                用于将 id ↔ token ↔ string 的 HuggingFace tokenizer。
            color : str, optional
                'red', 'green', 'yellow' ... 任选；决定 ANSI 颜色码。

            Returns
            -------
            str
                带 ANSI 颜色码的可打印字符串。
            """
            # ① 颜色表（可自行扩充）
            COLORS = {
                "red"   : "\033[91m",
                "green" : "\033[92m",
                "yellow": "\033[93m",
                "blue"  : "\033[94m",
                "magenta": "\033[95m",
                "cyan"  : "\033[96m",
            }
            C = COLORS.get(color.lower(), COLORS["red"])
            RESET = "\033[0m"

            # ② 保证都是可迭代对象
            ids  = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            mask = loss_mask.tolist() if hasattr(loss_mask, "tolist") else list(loss_mask)

            assert len(ids) == len(mask), "input_ids 与 loss_mask 长度必须一致"

            # ③ 将 id → token
            tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)

            # ④ 逐 token 组装文本
            pieces = []
            for tok, m in zip(tokens, mask):
                # convert_tokens_to_string 负责处理“前导空格/Ġ/▁”等特殊前缀
                text = tokenizer.convert_tokens_to_string([tok])
                if m:
                    pieces.append(f"{C}{text}{RESET}")
                else:
                    pieces.append(text)

            return "".join(pieces)




        chat_length = chat_ids.shape[0]

        input_ids = chat_ids
        #attention_mask = chat_attention_mask



        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            chat_attention_mask = torch.cat((chat_attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                chat_attention_mask = chat_attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                chat_attention_mask = chat_attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)
        #print(attention_mask.sum(), chat_attention_mask.sum())
        loss_mask = attention_mask.clone()


        assert loss_mask.shape == chat_attention_mask.shape






        return {
            'input_ids': input_ids,
            'attention_mask': chat_attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
