# scripts/code_editor.py
from pathlib import Path
import difflib
from config.openai_client import client, DEFAULT_MODEL, ROOT

def load_code(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def save_code(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def load_code_system_prompt() -> str:
    p = ROOT / "config" / "prompts" / "code_edit_system_prompt.md"
    return p.read_text(encoding="utf-8")

def call_model_for_code_edit(file_content: str, instruction: str) -> str:
    system_prompt = load_code_system_prompt()
    user_input = f"""下面是一个代码文件的完整内容，请根据“修改要求”返回修改后的完整文件。

### 修改要求
{instruction}

### 原始文件内容
```TEXT
{file_content}
```"""

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        max_output_tokens=64000,
        reasoning={"effort": "medium"},
    )
    return resp.output_text

def print_local_diff(old: str, new: str, context_lines: int = 5):
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))

    # 简单打印 unified diff（你也可以在这里进一步解析，只展示修改块）
    print("===== Unified diff =====")
    for line in diff:
        print(line)

    # 也可以之后做更复杂：找到 @@ -a,b +c,d @@ 块，抽取上下文 5 行等

def edit_file(path: str, instruction: str):
    p = Path(path)
    old_code = load_code(p)
    new_code = call_model_for_code_edit(old_code, instruction)

    # 先打印 diff，确认无误后再覆盖文件（你可以加一个确认 step）
    print_local_diff(old_code, new_code, context_lines=5)

    # 如果你希望自动覆盖，可以：
    save_code(p, new_code)
    print(f"[code_editor] file updated: {p}")

if __name__ == "__main__":
    # 示例：修改某个 C++ 文件
    edit_file(
        "workspace/projects/mesh_lab/code/HalfedgeMesh.cpp",
        "在 simplify 函数中加入检查：如果 collapse 之后违反 manifold 性，则跳过这条边。"
    )
