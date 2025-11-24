# scripts/summarize_paper.py
import json
from pathlib import Path
from config.openai_client import client, CHEAP_MODEL, ROOT, DATA

SUMMARY_DIR = DATA / "paper_summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def summarize_one_paper(paper_id: int, max_output_tokens: int = 6000):
    index_path = DATA / "paper_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    meta = next((m for m in index if m["id"] == paper_id), None)
    if meta is None:
        raise ValueError(f"paper id {paper_id} not found")

    txt_path = ROOT / meta["file_txt"]
    text = txt_path.read_text(encoding="utf-8")

    system_prompt = (
        "你是一个严谨的数学/机器学习论文阅读助手，擅长写技术向的中文综述。"
        "请为输入论文写一篇 3-5k tokens 左右的技术摘要，重点包括："
        "1) 问题设定与符号；2) 关键假设（例如高斯等价、独立性等）；"
        "3) 主定理/主结果；4) 证明或推导的整体结构；5) 与我的项目（如 DMFT、RL）可能相关的部分。"
        "最后留出一个“我的备注”部分，方便我自己补充理解。"
    )

    user_prompt = f"""下面是论文的纯文本内容（可能包含公式丢失或格式问题）：

{text}"""

    resp = client.responses.create(
        model=CHEAP_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    summary = resp.output_text

    # 在末尾加一个你可以编辑的备注块
    summary += "\n\n---\n\n## 我的备注（手动编辑此处）\n\n- "

    # 保存为 markdown
    safe_title = meta["title"].replace(" ", "_")[:40]
    out_path = SUMMARY_DIR / f"{meta['id']}_{safe_title}.md"
    out_path.write_text(summary, encoding="utf-8")
    print(f"[summary] saved to {out_path}")

if __name__ == "__main__":
    # 示例：为论文 id=1 生成摘要
    summarize_one_paper(1)

