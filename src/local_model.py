"""
Qwen3-0.6B 封装：汉字级概率提取

核心思路：
  Qwen3 的 tokenizer 中，绝大多数常用汉字是独立 token。
  对于每个测试位置：
    1. 取模型 logits
    2. 掩码：只保留单汉字 token
    3. Softmax → 汉字上的条件概率分布

用法:
  from src.local_model import CharModel
  model = CharModel()
  chars, probs = model.predict("今天天气")
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "pj/models/Qwen3-0.6B"


class CharModel:
    def __init__(self, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, device_map=device
        )
        self.device = next(self.model.parameters()).device
        self.char_ids, self.char_tokens = self._build_char_map()

    def _build_char_map(self):
        """扫描词表，找出所有单汉字 token 的 id 和对应字符"""
        ids, tokens = [], []
        for i in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode([i])
            if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
                ids.append(i)
                tokens.append(token)
        self._mask = torch.tensor(ids, device=self.device)
        return ids, tokens

    @torch.no_grad()
    def predict(self, text: str):
        """输入前缀，返回 (汉字列表, 概率列表)，按概率降序"""
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits[0, -1, :]

        # 掩码 + 重归一化
        masked = logits[self._mask]
        probs = torch.softmax(masked, dim=-1)

        # 排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        chars = [self.char_tokens[i] for i in sorted_indices.tolist()]
        return chars, sorted_probs.tolist()

    def predict_topk(self, text: str, k: int = 10):
        """取 top-K 预测"""
        chars, probs = self.predict(text)
        return chars[:k], probs[:k]
