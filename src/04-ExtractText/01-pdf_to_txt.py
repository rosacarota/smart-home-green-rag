from pathlib import Path
import argparse
import fitz  # PyMuPDF
import json
import re
import sys

PDF_DIR = Path("data/articles/pdfs")
TEXT_DIR = Path("data/articles/extracted_text")
REPORT_DIR = Path("data/articles/extraction_reports")

TEXT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    # Merge words split across line breaks: "ener-\ngy" -> "energy"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace multiple consecutive line breaks with a double line break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def quality_report(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text)
    chars = len(text)

    # Simple heuristic: count how many non-empty lines are very short
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = [line for line in lines if len(line) < 35]

    return {
        "num_chars": chars,
        "num_words": len(words),
        "num_lines": len(lines),
        "short_line_ratio": round(len(short_lines) / len(lines), 3) if lines else 1.0,
        "looks_empty": chars < 2000,
        "may_be_bad_extraction": (
            chars < 2000
            or (len(lines) > 0 and len(short_lines) / len(lines) > 0.65)
        ),
    }


def extract_pdf_text(pdf_path: Path) -> str:
    pages_text = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages_text.append(f"\n\n--- PAGE {page_number} ---\n\n{text}")

    return clean_text("\n".join(pages_text))


def normalize_article_name(article: str) -> str:
    # Accept both "4" and "article_4" as valid input formats
    article = article.strip()

    if article.startswith("article_"):
        return article

    return f"article_{article}"


def find_pdf_for_article(article: str) -> Path:
    article_name = normalize_article_name(article)

    # Try the direct filename first, for example: article_4.pdf
    direct_path = PDF_DIR / f"{article_name}.pdf"
    if direct_path.exists():
        return direct_path

    # Try a case-insensitive search as a fallback
    matching_files = [
        pdf_path
        for pdf_path in PDF_DIR.glob("*.pdf")
        if pdf_path.stem.lower() == article_name.lower()
    ]

    if matching_files:
        return matching_files[0]

    available = sorted(pdf_path.stem for pdf_path in PDF_DIR.glob("*.pdf"))

    raise FileNotFoundError(
        f"PDF not found for article '{article}'.\n"
        f"Expected file: {direct_path}\n"
        f"Available articles: {available}"
    )


def parse_single_pdf(pdf_path: Path) -> None:
    article_id = pdf_path.stem

    text = extract_pdf_text(pdf_path)
    report = quality_report(text)

    txt_path = TEXT_DIR / f"{article_id}.txt"
    report_path = REPORT_DIR / f"{article_id}_report.json"

    txt_path.write_text(text, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "⚠️ check manually" if report["may_be_bad_extraction"] else "✅ ok"

    print(f"{article_id} {status}")
    print(f"Text saved to: {txt_path}")
    print(f"Report saved to: {report_path}")
    print(f"Report: {report}")


def parse_all_pdfs() -> None:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    for pdf_path in pdf_files:
        print("-" * 80)
        parse_single_pdf(pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from one or more article PDFs."
    )

    parser.add_argument(
        "--article",
        type=str,
        help="Article number or name to parse, for example: 4 or article_4.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Parse all PDF files in the input folder.",
    )

    args = parser.parse_args()

    if not PDF_DIR.exists():
        print(f"PDF directory not found: {PDF_DIR}")
        sys.exit(1)

    if args.all and args.article:
        print("Please use either --all or --article, not both.")
        sys.exit(1)

    if args.all:
        parse_all_pdfs()
        return

    if args.article:
        pdf_path = find_pdf_for_article(args.article)
        parse_single_pdf(pdf_path)
        return

    print("No article selected.")
    print("Use one of the following commands:")
    print("  python extract_pdf_text.py --article 4")
    print("  python extract_pdf_text.py --article article_4")
    print("  python extract_pdf_text.py --all")


if __name__ == "__main__":
    main()