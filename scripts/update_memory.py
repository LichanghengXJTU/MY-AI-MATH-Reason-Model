# scripts/update_memory.py
from pathlib import Path
import json
from config.openai_client import client, CHEAP_MODEL, ROOT, WS

def update_project_memory(project: str, max_tokens: int = 2000):
    proj_dir = WS / project
    hist_path = proj_dir / "history.jsonl"
    mem_path = proj_dir / "memory.md"

    if not hist_path.exists():
        print("no history yet")
        return

    records = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines()]
    # 取最近若干次的 note 内容
    texts = []
    for r in records[-5:]:
        p = ROOT / r["answer_path"]
        texts.append(p.read_text(encoding="utf-8"))

    prompt = (
        "你是这个项目的记录员。下面是最近几次 AI 推导的内容，"
        "请用中文整理出一个项目级别的记忆摘要，突出：已确定的定义、重要结论、尚未解决的问题。\n\n"
        + "\n\n-----\n\n".join(texts)
    )

    resp = client.responses.create(
        model=CHEAP_MODEL,
        input=prompt,
        max_output_tokens=max_tokens,
    )

    summary = resp.output_text
    mem_path.write_text(summary, encoding="utf-8")
    print(f"[memory] updated {mem_path}")

