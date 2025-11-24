# scripts/ingest_paper.py
import json
from pathlib import Path
import PyPDF2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "papers_raw"
TXT = ROOT / "data" / "papers_text"
INDEX = ROOT / "data" / "paper_index.json"

def pdf_to_text(pdf_path: Path) -> str:
    reader = PyPDF2.PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n\n".join(texts)

def load_index():
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return []

def save_index(idx):
    INDEX.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")

def ingest_one(pdf_path: Path, title: str = None, tags=None):
    tags = tags or []
    text = pdf_to_text(pdf_path)

    txt_name = pdf_path.stem + ".txt"
    txt_path = TXT / txt_name
    TXT.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")

    idx = load_index()
    paper_id = len(idx) + 1
    idx.append({
        "id": paper_id,
        "file_pdf": str(pdf_path.relative_to(ROOT)),
        "file_txt": str(txt_path.relative_to(ROOT)),
        "title": title or pdf_path.stem,
        "tags": tags,
    })
    save_index(idx)
    print(f"[ingest] {pdf_path.name} -> id={paper_id}")

if __name__ == "__main__":
    # 示例：批量导入 data/papers_raw 下的 PDF
    RAW.mkdir(parents=True, exist_ok=True)
    for pdf in RAW.glob("*.pdf"):
        ingest_one(pdf)
