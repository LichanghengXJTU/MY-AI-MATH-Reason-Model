# scripts/math_agent.py
import json
import sys
from pathlib import Path
from datetime import datetime

# 把项目根目录加入 sys.path，方便从 scripts/ 看到 config/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.openai_client import client, DEFAULT_MODEL


DATA = ROOT / "data"
WS = ROOT / "workspace" / "projects"

from typing import Tuple

def load_project_documents(project: str) -> Tuple[str, str]:
    """
    尝试加载该项目下的论文草稿和 proposal。
    如果对应文件不存在，就返回空字符串。
    """
    proj_dir = WS / project / "paper"
    draft = ""
    proposal = ""

    draft_path = proj_dir / "main.tex"
    if draft_path.exists():
        draft = draft_path.read_text(encoding="utf-8")

    prop_path = proj_dir / "proposal.tex"
    if prop_path.exists():
        proposal = prop_path.read_text(encoding="utf-8")

    return draft, proposal


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_math_system_prompt() -> str:
    p = ROOT / "config" / "prompts" / "math_system_prompt.md"
    return load_text(p)

def load_project_memory(project: str) -> str:
    mem_path = WS / project / "memory.md"
    if mem_path.exists():
        return mem_path.read_text(encoding="utf-8")
    return ""

def append_history(project: str, record: dict):
    proj_dir = WS / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    hist_path = proj_dir / "history.jsonl"
    with hist_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_note(project: str, title_slug: str, content: str) -> Path:
    notes_dir = WS / project / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = notes_dir / f"{ts}_{title_slug}.md"
    path.write_text(content, encoding="utf-8")
    return path

def build_user_input(question: str, paper_ids: list[int]) -> str:
    """
    把问题 + 若干论文的“摘要（若有）/全文”拼起来。
    如果存在 data/paper_summaries/{id_*}.md，就优先使用摘要。
    """
    idx_path = DATA / "paper_index.json"
    index = json.loads(idx_path.read_text(encoding="utf-8"))

    parts = [f"### 当前任务\n{question}\n"]
    parts.append("### 可用论文信息（摘要优先）\n")

    summary_dir = DATA / "paper_summaries"

    for meta in index:
        if meta["id"] in paper_ids:
            header = f"[PAPER {meta['id']}: {meta['title']}]"

            # 先找摘要
            summary_file = None
            for p in summary_dir.glob(f"{meta['id']}_*.md"):
                summary_file = p
                break

            if summary_file and summary_file.exists():
                text = summary_file.read_text(encoding="utf-8")
                parts.append(header + "（以下为技术摘要 + 我的备注）\n" + text + "\n")
            else:
                # 回退到全文
                txt_path = ROOT / meta["file_txt"]
                text = txt_path.read_text(encoding="utf-8")
                parts.append(header + "（未生成摘要，以下为全文）\n" + text + "\n")

    return "\n\n".join(parts)


def ask_math(
    project: str,
    question: str,
    paper_ids: list[int],
    max_output_tokens: int = 40000,
):
    system_prompt = load_math_system_prompt()
    memory = load_project_memory(project)
    user_input = build_user_input(question, paper_ids)

    draft, proposal = load_project_documents(project)

    paper_context = ""
    if draft:
        paper_context += "### 当前论文草稿（仅供参考，不要求你重写论文）\n"
        paper_context += (
            "下面是当前的 LaTeX 论文草稿节选。"
            "你的推导应该尽量与其中的符号和记号保持一致，"
            "并在适当位置标注“可插入到论文第几节”。\n\n"
            + draft
        )
    if proposal:
        paper_context += "\n\n### 当前 Proposal（研究目标和约束）\n"
        paper_context += (
            "下面是当前 proposal 的内容。你进行推导时要尊重这里的研究目标，"
            "如果发现 proposal 中的假设与推导结果冲突，请明确指出。\n\n"
            + proposal
        )

    full_input = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n下面是该项目的已有记忆（可参考）：\n"
                + memory
            ),
        },
        {
            "role": "user",
            "content": (
                "下面依次给出：\n"
                "1. 本次任务说明\n"
                "2. 相关论文片段\n"
                "3. 当前论文草稿和 proposal\n\n"
                "==== 1. 本次任务 ====\n"
                + user_input
                + "\n\n==== 3. 论文草稿与 Proposal ====\n"
                + paper_context
            ),
        },
    ]


    print("[math_agent] calling gpt-5.1 ...")
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=full_input,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "high"},  # 让它认真想
    )

    answer = resp.output_text  # 官方 SDK 提供的快捷属性 :contentReference[oaicite:6]{index=6}

    record = {
        "ts": datetime.now().isoformat(),
        "project": project,
        "model": DEFAULT_MODEL,
        "question": question,
        "paper_ids": paper_ids,
        "input_tokens": resp.usage.input_tokens if hasattr(resp, "usage") else None,
        "output_tokens": resp.usage.output_tokens if hasattr(resp, "usage") else None,
        "answer_path": None,
    }

    note_path = save_note(project, "math_derivation", answer)
    record["answer_path"] = str(note_path.relative_to(ROOT))
    append_history(project, record)

    print(f"[math_agent] answer saved to {note_path}")
    return answer

def retry_last(project: str, critique: str):
    proj_dir = WS / project
    hist_path = proj_dir / "history.jsonl"
    records = [json.loads(line) for line in hist_path.read_text(encoding="utf-8").splitlines()]
    last = records[-1]

    old_answer = (ROOT / last["answer_path"]).read_text(encoding="utf-8")

    user_input = f"""下面是我上一次要求你解决的任务原问题，以及你给出的推导结果。
我现在对这个结果不满意，原因是：{critique}

请在保留有价值部分的基础上，重新组织和改写推导。注意：
- 必须完整给出关键步骤，而不是只给结论。
- 对之前存在的问题要明确指出并修正。

### 原问题
{last['question']}

### 你上一次的回答
{old_answer}
"""

    system_prompt = load_math_system_prompt()
    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        reasoning={"effort": "high"},
        max_output_tokens=20000,
    )
    answer = resp.output_text
    note_path = save_note(project, "retry_derivation", answer)
    # 也可以追加到 history.jsonl
    print(f"[retry] new answer saved to {note_path}")
    return answer

def ask_from_prompt_file(project: str, prompt_path: str, paper_ids: list[int]):
    prompt_file = WS / project / "prompts" / prompt_path
    question = prompt_file.read_text(encoding="utf-8")
    return ask_math(project, question, paper_ids)



if __name__ == "__main__":
    project = "Loss_Dynamics_in_Temporal_Differences_of_Reinforcement_Learning_with_Unfixed_Policy"
    paper_ids = [1]

    ans = ask_from_prompt_file(
        project,
        "0.md",
        paper_ids
    )
    print(ans[:2000])

