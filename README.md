# AI Math Lab 使用说明（API 调用器与高输出 Token 配置）

本项目提供一个“数学研究/推导”工作流，围绕一个简单的 API 调用器（`scripts/math_agent.py`），结合论文文本与项目草稿，产出尽可能长且结构化的推导结果。本文档将指导你：

- 如何安装依赖、配置密钥与环境
- 如何放置论文 PDF/文本与项目代码/草稿
- 如何导入论文并建立索引
- 如何运行推导并配置“最大输出 tokens”，尽量接近模型的上限
- 常见问题与排错

---

## 1. 快速开始

1) 安装 Python 版本（建议 3.10+）

2) 安装依赖（推荐使用虚拟环境）

```bash
pip install -U openai python-dotenv PyPDF2
```

3) 准备 OpenAI API Key

- 在 `config/.env` 写入：
  ```ini
  OPENAI_API_KEY=你的API密钥
  ```
- 或者在 Shell 里临时导出：
  ```bash
  export OPENAI_API_KEY="你的API密钥"
  ```

4) 推荐安装 pdftotext（Poppler）

- macOS:
  ```bash
  brew install poppler
  ```
- 安装后应可直接使用 `pdftotext` 命令。

---

## 2. 项目结构与放置规范

本仓库关键目录如下（只列与用户相关的）：

```
ai-math-lab/
  config/
    .env                      # 存放 OPENAI_API_KEY（不提交到仓库）
    openai_client.py          # OpenAI 客户端与默认模型
    prompts/
      math_system_prompt.md   # 数学推导系统提示词（会作为 system role）
  data/
    paper_index.json          # 论文索引（id、标题、txt 路径等）
    papers_raw/               # 放 PDF 原件（可选）
    papers_text/              # 放 .txt（由 pdftotext 或脚本生成）
    paper_summaries/          # 论文技术摘要（可选，用以节省输入 tokens）
  scripts/
    ingest_paper.py           # 导入 PDF，提取文本并更新索引
    summarize_paper.py        # 生成论文技术摘要（可选）
    math_agent.py             # API 调用器：执行推导与保存结果
  workspace/
    projects/
      <YourProject>/
        code/                 # 你的项目代码（便于集中管理，可选）
        paper/
          main.tex            # 你的论文草稿（可选，但强烈建议提供）
          proposal.tex        # 你的研究 Proposal（可选）
        memory.md             # 项目记忆/约定（可选）
        prompts/
          0.md                # 任务描述/问题（math_agent 示例会读取此文件）
        notes/
          <timestamp>_math_derivation.md  # 推导结果自动保存到此
        history.jsonl         # 每次调用的元信息日志
```

放置要点：
- 论文 PDF：放到 `data/papers_raw/`
- 论文 TXT：强烈建议用 `pdftotext` 生成后放到 `data/papers_text/`
- 项目论文草稿：放到 `workspace/projects/<YourProject>/paper/main.tex`
- 项目 Proposal：放到 `workspace/projects/<YourProject>/paper/proposal.tex`
- 项目“记忆/约定”：放到 `workspace/projects/<YourProject>/memory.md`
- 任务描述：放到 `workspace/projects/<YourProject>/prompts/0.md`
- 你的代码：统一放 `workspace/projects/<YourProject>/code/`（供你组织管理，当前调用器不会直接读取代码内容）

---

## 3. 论文导入与索引

### 3.1 推荐方式：先用 pdftotext，再更新索引

1) 将 PDF 放入 `data/papers_raw/`  
2) 使用 `pdftotext` 生成 `.txt`：
```bash
pdftotext "data/papers_raw/YourPaper.pdf" "data/papers_text/YourPaper.txt"
```
3) 更新 `data/paper_index.json`，添加一条记录（自增 id，示例）：
```json
[
  {
    "id": 1,
    "file_pdf": "data/papers_raw/YourPaper.pdf",
    "file_txt": "data/papers_text/YourPaper.txt",
    "title": "YourPaper",
    "tags": ["RL", "TD"]
  }
]
```
提示：
- 文件路径最好使用相对仓库根目录的相对路径（项目脚本即按此读取）
- `id` 必须唯一，后续在调用器中用 `paper_ids` 引用

### 3.2 备选方式：使用脚本自动提取

你也可以把 PDF 放在 `data/papers_raw/` 后，直接运行：
```bash
python scripts/ingest_paper.py
```
它会用 `PyPDF2` 提取文本并写入 `data/papers_text/`，同时自动在 `data/paper_index.json` 里追加记录。  
注意：对于公式密集或排版复杂的 PDF，`pdftotext` 的质量通常优于 `PyPDF2`，建议优先使用 3.1 的流程。

### 3.3 可选：生成技术摘要节省输入 tokens

如果准备多篇论文，建议先生成“技术摘要”，调用器会优先使用摘要，显著减少输入 tokens：
```bash
python -c "from scripts.summarize_paper import summarize_one_paper; summarize_one_paper(1, max_output_tokens=8000)"
```
生成结果会保存到 `data/paper_summaries/`，`scripts/math_agent.py` 在拼装输入时会优先读取该目录下与论文 id 对应的摘要文件。

---

## 4. 运行推导（API 调用器）

核心脚本：`scripts/math_agent.py`

它会做三件事：
1) 读取系统提示词 `config/prompts/math_system_prompt.md`（作为 system role）
2) 拼装用户输入：你的任务说明（`prompts/0.md`）+ 论文摘要或全文（来自 `paper_index.json`）+ 论文草稿/Proposal（若存在）
3) 调用模型并将结果保存到 `workspace/projects/<YourProject>/notes/<timestamp>_math_derivation.md`，同时在 `history.jsonl` 记录元信息（包括输入/输出 tokens）

### 4.1 最快验证路径（直接运行示例）

编辑 `scripts/math_agent.py` 末尾的示例参数：
```python
if __name__ == "__main__":
    project = "Loss_Dynamics_in_Temporal_Differences_of_Reinforcement_Learning_with_Unfixed_Policy"
    paper_ids = [1]

    ans = ask_from_prompt_file(
        project,
        "0.md",
        paper_ids
    )
    print(ans[:2000])
```
将 `project` 替换成你的项目名（需提前建好目录与 `prompts/0.md`），把 `paper_ids` 改成你在 `paper_index.json` 里配置的论文 id 列表，然后运行：
```bash
python scripts/math_agent.py
```

### 4.2 以函数方式调用并控制“最大输出 tokens”

你也可以在交互式环境里直接调用：
```python
from scripts.math_agent import ask_math

answer = ask_math(
    project="YourProject",
    question="请围绕 TD 学习中的 deadly triad 给出严谨推导与条件边界……",
    paper_ids=[1, 3],
    max_output_tokens=40000  # 根据你的模型上限调整
)
print(answer[:2000])
```
`max_output_tokens` 决定了本次回答可输出的最大 tokens 数；如果你拥有更高上限的模型，可以把该参数调大（详见下一节）。

---

## 5. 如何“尽量接近模型上限的输出 tokens”

本项目采用官方 SDK 的 `responses.create`，由 `config/openai_client.py` 指定默认模型：

```python
DEFAULT_MODEL = "gpt-5.1"
CHEAP_MODEL = "gpt-5-mini"
```

在 `scripts/math_agent.py` 的 `ask_math()` 中，你可以传入：
```python
resp = client.responses.create(
    model=DEFAULT_MODEL,
    input=full_input,
    max_output_tokens=max_output_tokens,
    reasoning={"effort": "high"},
)
```

要让输出尽量接近模型上限，请同时注意：
- 调大 `max_output_tokens`：调用 `ask_math(..., max_output_tokens=...)`，填到你所用模型允许的最大值或接近最大值。
- 控制总上下文长度：总 tokens = 输入 tokens + 输出 tokens ≤ 模型上限。  
  建议：
  - 优先为每篇论文生成“技术摘要”，调用器会优先拼入摘要，极大降低输入体积（目录：`data/paper_summaries/`）
  - 只选择本次任务强相关的论文 id（例如 `paper_ids=[2,5]` 而非一口气塞入十几篇）
  - 精简 `workspace/projects/<YourProject>/paper/main.tex` 中不相关的大段文本或只保留对齐符号的必要片段
  - 精简 `memory.md` 的无关内容，保留术语规范/关键约定
- 选择高上限模型：如你有更大上下文/输出上限的模型权限，请在 `config/openai_client.py` 将 `DEFAULT_MODEL` 改为相应模型。
- 延长超时：`openai_client.py` 中 `timeout=3000`（秒）已较大，长答案更稳。

调用完成后，可在：
- `workspace/projects/<YourProject>/notes/` 查看完整推导稿
- `workspace/projects/<YourProject>/history.jsonl` 查看输入/输出 tokens 统计

---

## 6. 提示词与风格

- 修改 `config/prompts/math_system_prompt.md` 可定制推导规范、结构与风格（例如是否更“教科书式”）
- 保持推导中的符号与 `paper/main.tex` 一致，调用器已将草稿与 Proposal 一并注入上下文

---

## 7. 常见问题与排错

- 没有输出或报 401：检查 `OPENAI_API_KEY` 是否正确加载（`.env` 或环境变量）
- 超时：在 `config/openai_client.py` 调整 `timeout`，或减少输入体积
- 输出长度不够：提高 `max_output_tokens`；减少输入 tokens（见第 5 节）
- 论文文本乱码/缺字：优先使用 `pdftotext` 生成 `.txt`，再更新 `paper_index.json`
- `paper_index.json` 找不到 id：确认 `id` 唯一且与 `paper_ids` 一致
- 生成摘要失败：确保 `data/papers_text/` 下对应 `.txt` 存在，且摘要长度上限通过参数 `max_output_tokens` 合理设置

---

## 8. 最佳实践清单

- 以“任务 → 相关论文摘要 → 草稿/Proposal”的顺序组织输入，避免无关材料
- 总是优先生成技术摘要，再进行长篇推导请求
- 在 `prompts/0.md` 中写清楚问题与期望产出格式（例如章节结构、需要的定理/边界条件）
- 保持 `history.jsonl` 与 `notes/` 的版本化，便于回溯与二次编辑

---