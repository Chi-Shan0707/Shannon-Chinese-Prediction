# 项目方案：用LLM模拟Shannon汉语预测实验

## 一、背景与问题

Shannon（1951）通过人类受试者逐字母预测英语文本，估算英语的N-gram熵和冗余度。
核心发现：已知前N个字母时，下一个字母的可预测性随N增大而提升；英语冗余度约75%。

### 汉语熵研究现状（文献核实，2026-05）

汉语熵的已有研究**全程依赖计算语料库方法**，从未做过Shannon式的人类逐字预测实验——汉语常用字约6700个，人类受试者无法在合理次数内猜中。

**字级（character-level）已有结果：**

| 阶数 | 方法 | 熵值 | 来源 |
|------|------|------|------|
| Unigram (H0) | 字频统计 | **9.62 bits/char** | 孙帆、孙茂松（《人民日报》语料库） |
| Bigram (H1) | 条件概率 | **~7.15 bits/char** | 孙帆、孙茂松 |
| Trigram (H2) | 条件概率 | **~6.65 bits/char** | 孙帆、孙茂松 |
| Class-based bigram | 同义词词林聚类 | **~4.6 bits/char** | Chang & Lin (1994), ACL Anthology |
| 神经网络LM | AWD-LSTM-MoS | **4.43 bpc**；外推至无穷 ~**3.96 bpc** | Takahashi & Tanaka-Ishii (2018), *Entropy* 20(11), 839 |

**词级（word-level）已有结果：**
- Unigram: ~13.7 bits/word
- Bigram 条件熵: ~6.4 bits/word
- Trigram 条件熵: ~1.5 bits/word

**关键发现：** Takahashi & Tanaka-Ishii (2018) 使用的AWD-LSTM-MoS是目前最紧的汉语熵率上界（3.96 bpc），但该方法是神经网络语言模型的交叉熵，而非Shannon式预测实验。**至今没有人构建过汉语的Shannon条件熵链（H0→H1→H2→...→H∞）。**

### 本项目填补的空白

用LLM代替人类受试者，首次对汉语做Shannon式预测实验，构建完整的条件熵链。对比：
- LLM vs 人类的预测能力差异
- 汉语 vs 英语的统计结构差异

---

## 二、三项关键问题的回答

### Q1：获取token概率的方式——API还是本地部署？

#### 选项A：API方案（推荐起步）

| 服务 | 接口 | top_logprobs最大数量 | 中文能力 | 费用 |
|------|------|---------------------|---------|------|
| OpenAI | chat.completions + logprobs | 5 | GPT-4o中文优秀 | 按token付费 |
| Together AI | chat.completions + logprobs | 5~10 | 接入Llama等开源模型 | 便宜 |
| DeepInfra | chat.completions + logprobs | 20 | 接入DeepSeek等模型 | 便宜 |
| Zhipu API | chat.completions | 可能支持 | GLM-4中文最强 | 你已经有关键 |

**关键限制**：top_logprobs 最多返回5~20个候选。中文常用字约6700个，
如果只取top-5，丢失大量概率质量，下界估计可能不准。

**解决方案**：使用 `logit_bias` 参数或取 `temperature=0` + `max_tokens=1` 
多次采样来估计概率分布（见选项B的变体）。

#### 选项B：本地部署方案（更完整但更重）

用 Hugging Face transformers + AutoModelForCausalLM 加载开源中文模型
（如 Qwen2.5、DeepSeek、Yi等），直接读取 `logits` 层输出：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

inputs = tokenizer("今天天气", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 100)  # 取前100个
```

**优势**：获取完整概率分布（整个vocabulary），不限于top-5/20
**劣势**：需要GPU资源（7B模型约14GB显存）

**推荐思路**：先用API做原型验证，再用本地模型做完整实验。

#### Q1b：取概率最大还是保留随机性？

Shannon实验的设计是**确定性预测**——受试者按最佳猜测依次尝试，
应该用 `temperature=0` + top_logprobs 取排名，模拟"第一次猜最可能、第二次猜第二可能"。

但如果想对比"LLM作为人类模拟器"的能力，可以**同时做两种**：
1. **确定性模式**：`temperature=0`，严格按概率排序猜测
2. **采样模式**：`temperature=1.0`，让LLM正常生成，
   对比其自然输出与确定性预测的差异

---

### Q2：汉语预测实验的形式

#### 汉语与英语的结构差异

| 维度 | 英语 | 汉语 |
|------|------|------|
| 基本单位 | 字母(26) | 汉字(常用6700) |
| 分词 | 有空格分隔 | 无天然词边界 |
| 零阶熵 | 4.03 bit | 9.71 bit |
| 冗余度(统计) | ~73% | ~63% |
| Shannon实验可行性 | 容易（猜字母） | 极难（猜汉字需几千次） |

#### 过往汉语熵研究的方式

汉语熵**从未**做过Shannon式的逐字人类预测实验，原因：
- 英语26个字母，平均4~5次猜中
- 汉语6700个常用字，平均需要 ~12次（零阶），实际上不可行

已有研究全部采用**语料库统计**方法：
- 北航（1980s）：统计字频计算零阶熵 9.71 bit
- 词熵计算：提取词频分布，计算一阶熵 ~11.46 bit
- N-gram语言模型：用Kneser-Ney平滑的三元模型估算条件熵
- Zipf定律检验：验证汉字/词频分布是否符合幂律

#### 本项目建议的实验设计

**设计思路**：不猜字，猜token（LLM的subword单位）。

但为了可比性，更好的方式是**两种粒度**：

**方案A：字级预测（推荐为主要方案）**
- Prefix：给定前N个汉字 → 预测下一个汉字
- 字母表大小：常用汉字 ~6700个（覆盖99%语料）
- 实验取前N个汉字 = 1, 2, 3, 5, 10, 20, 50
- 每次预测按概率排序取Top-K
- 计算条件熵 H(chinese_char | prev_N_chars)

**方案B：词级预测（辅助方案）**
- 先用分词工具（jieba/Stanford分词）切词
- 给定前N个词 → 预测下一个词
- 与Shannon原文的"Fword"近似对比

**语料选择建议**：
- 现代汉语平衡语料库（如人民日报语料）
- 文学作品选段（类似于Shannon用的"Jefferson the Virginian"）
- 至少100个测试样本，每个15~50字

---

### Q3：与英语实验的差异总结

| 维度 | Shannon英语 | 本项目汉语 |
|------|-----------|-----------|
| 预测单位 | 字母(26) | 汉字(6700) / token |
| 受试者 | 人类(Shannon妻子) | LLM |
| 测试数 | 100样本×15字母 | 建议100+样本×15~50字 |
| 数据收集 | 人工逐次猜测 | API调用/top_logprobs |
| N范围 | 0~15, 100 | 建议0~50 |
| 计算方法 | Shannon不等式(17) | 同款不等式 or 直接概率求和 |

---

## 三、实验方案细节

### 3.1 数据准备

```
data/
  ├── chinese_samples.txt    # 从语料库选取的测试样本
  ├── character_frequency.txt # 汉字频率表（用于零阶熵基线）
  └── references/            # 参考文献
```

选取标准：
- 现代白话文
- 覆盖新闻、文学、科普等多种语体（与Shannon做法一致）
- 每段长度>50字
- 至少100个样本

### 3.2 实验流程

1. **零阶熵 H0**：用字频表计算（已有：9.71 bit）
2. **一阶熵 H1**：给定前1个字，预测下一个字
3. **N阶熵 HN**：给定前 N 个字，预测下一个字
   - N = 1, 2, 3, 5, 10, 20, 50
4. **对每个测试位置**：
   - 构造 prompt："以下句子的下一个字是什么？\n句子：xxxxx\n下一个字："
   - 通过API或本地模型获取 logprobs/top_logprobs
   - 记录完整概率分布（或top-K分布）
5. **计算上下界**：
   - 上界：-Σ q_i log q_i
   - 下界：Σ i(q_i - q_{i+1}) log i

### 3.3 LLM策略选项

| 选项 | 模型 | 部署 | 优势 | 劣势 |
|------|------|------|------|------|
| A1 | GPT-4o (API) | OpenAI | logprobs现成，中文强 | top_logprobs=5不够 |
| A2 | DeepSeek-V3 (API) | DeepInfra/Together | top_logprobs=20 | logprobs数多但非完整分布 |
| A3 | GLM-4 (API) | Zhipu | 中文最强，你有key | 需确认logprobs支持 |
| B1 | Qwen2.5-7B (本地) | HuggingFace | 完整概率分布 | 需要GPU(~14GB) |
| B2 | Qwen2.5-32B (量化) | 本地/HF | 效果更好，完整分布 | 更大显存 |

**推荐路径**：
- Phase 1：用API（GLM-4 或 GPT-4o）做小规模原型（20样本 × 5个N值），验证流程
- Phase 2：用本地Qwen2.5做完整实验（100样本 × 7个N值），获取完整概率分布

### 3.4 预期结果

| N | 条件熵(汉语) | 条件熵(Shannon英语) | 说明 |
|---|-------------|-------------------|------|
| 0 | ~9.7 bit | 4.7 bit | 零阶，等概率假设 |
| 1 | ~7-8 bit | 4.1 bit | 已知前1个字/字母 |
| 2 | ~6-7 bit | 3.6 bit | 已知前2个字/字母 |
| 3 | ~5-6 bit | 3.3 bit | 已知前3个字/字母 |
| 5 | ~4-5 bit | ~2.7 bit | 已知前5个 |
| 10 | ~3-4 bit | ~2.1 bit | 已知前10个 |
| 50 | ~2-3 bit | ~1.3 bit | 已知前50个 |
| ∞(外推) | ~2 bit | ~1 bit | 极限熵 |

对比预期：汉语每字信息量大约是英语每字母的2倍左右，
但汉语一字的冗余度（9.7bit字的63%≈6.1bit冗余）
比英语一根字母（4.7bit的75%≈3.5bit冗余）绝对值更高。

### 3.5 创新点

1. **首次用LLM模拟Shannon汉语预测实验**
2. **对比LLM vs 人类的预测策略差异**（Shannon仅有他妻子一个受试者）
3. **对比汉语与英语的统计结构差异**（基于信息论的统一框架）
4. **验证LLM的概率输出是否与人类统计直觉一致**

---

## 四、文件夹结构

```
pj/
├── shannon_51.pdf                           # Shannon原始论文
├── pdfplumber_text.txt                      # pdfplumber提取的文字
├── ocr_full_text.txt                        # GLM-4V OCR文字
├── comparison_analysis.txt                  # 两种提取方式对比
│
├── plans/
│   ├── project_plan.md                      # 本文件（完整方案）
│   └── experiment_design.md                 # 实验设计细节
│
├── data/
│   ├── chinese_samples.txt                  # 测试样本
│   ├── character_frequency.txt              # 汉字频率表
│   ├── sample_selection.py                  # 样本选取脚本
│   └── references/
│       └── related_papers.md                # 相关文献索引
│
├── src/
│   ├── api_clients/
│   │   ├── openai_client.py                 # OpenAI API客户端
│   │   └── zhipu_client.py                  # 智谱API客户端
│   ├── local_model.py                       # 本地HuggingFace模型接口
│   ├── experiment.py                        # 预测实验主逻辑
│   ├── entropy_calculator.py                # 熵值计算（上下界）
│   └── utils.py                             # 工具函数
│
├── reports/
│   ├── results_summary.md                   # 实验结果汇总
│   └── figures/                             # 图表
│
├── ocr_output/                              # OCR中间结果
│   └── page_*.txt / page_*.png
│
└── archive/                                 # 废弃/暂存文件
```

---

## 五、路线图

### Phase 0：设置环境（~1天）
- [ ] 确定LLM提供商与API
- [ ] 整理语料样本
- [ ] 确认实验协议的可行性

### Phase 1：原型验证（~2天）
- [ ] 用API调通实验流程
- [ ] 20样本 × 5个N值
- [ ] 验证Shannon不等式上下界计算逻辑

### Phase 2：完整实验（~3天）
- [ ] 100+样本 × 7个N值
- [ ] 两种模型/策略对比
- [ ] 生成图表和数据

### Phase 3：分析撰写（~2天）
- [ ] LLM vs 人类对比
- [ ] 汉语 vs 英语对比
- [ ] 撰写论文/报告

---

## 六、参考资源

1. Shannon (1951). Prediction and Entropy of Printed English. BSTJ.
2. Shannon (1948). A Mathematical Theory of Communication. BSTJ.
3. 北航汉字频率统计（1980s）→ 汉字零阶熵9.71 bit
4. 汉语的熵及其在语言本体研究中的应用（博士论文）
5. Bentz et al. (2017). The Entropy of Words — 跨1000+语言的熵研究
6. OpenAI API Logprobs文档
7. Chen & Goodman (1999). An empirical study of smoothing techniques for language modeling.
