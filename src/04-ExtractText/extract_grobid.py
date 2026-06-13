from pathlib import Path
import argparse
import inspect
import shutil
import tempfile
import xml.etree.ElementTree as ET

from grobid_client.grobid_client import GrobidClient


PDF_DIR = Path("data/articles/pdfs")
OUTPUT_DIR = Path("data/articles/extracted_grobid_tei")

GROBID_SERVER = "http://localhost:8070"
GROBID_TIMEOUT_SECONDS = 300

TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NAMESPACE}

# Request coordinates only for elements useful during document cleaning.
# Tables are normally represented by GROBID as <figure type="table">.
GROBID_COORDINATE_ELEMENTS = [
    "title",
    "head",
    "p",
    "s",
    "note",
    "figure",
    "formula",
    "ref",
]


def normalize_article_name(article: str) -> str:
    article = article.strip()

    if article.lower().endswith(".pdf"):
        article = article[:-4]

    if article.startswith("article_"):
        return article

    return f"article_{article}"


def find_pdf(article: str) -> Path:
    article_name = normalize_article_name(article)
    pdf_path = PDF_DIR / f"{article_name}.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    return pdf_path


def find_all_pdfs() -> list[Path]:
    pdf_files = sorted(PDF_DIR.glob("article_*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No article PDF files found in: {PDF_DIR}"
        )

    return pdf_files


def create_client() -> GrobidClient:
    """
    Create a GROBID client while remaining compatible with slightly
    different grobid-client-python versions.
    """
    constructor_parameters = inspect.signature(
        GrobidClient
    ).parameters

    client_arguments: dict[str, object] = {
        "grobid_server": GROBID_SERVER,
    }

    if "coordinates" in constructor_parameters:
        client_arguments["coordinates"] = (
            GROBID_COORDINATE_ELEMENTS
        )

    if "timeout" in constructor_parameters:
        client_arguments["timeout"] = (
            GROBID_TIMEOUT_SECONDS
        )

    if "check_server" in constructor_parameters:
        client_arguments["check_server"] = True

    return GrobidClient(**client_arguments)


def set_supported_argument(
    process_arguments: dict[str, object],
    process_parameters: dict,
    aliases: tuple[str, ...],
    value: object,
    required: bool = False,
) -> str | None:
    """
    Add an argument using the first parameter name supported by the
    installed GROBID client.

    Some client versions use snake_case while others expose names
    closer to the REST API.
    """
    for alias in aliases:
        if alias in process_parameters:
            process_arguments[alias] = value
            return alias

    if required:
        joined_aliases = ", ".join(aliases)

        raise RuntimeError(
            "The installed GROBID client does not support any of "
            f"the expected parameters: {joined_aliases}"
        )

    return None


def build_process_arguments(
    client: GrobidClient,
    input_dir: Path,
    workers: int,
) -> dict[str, object]:
    process_parameters = inspect.signature(
        client.process
    ).parameters

    process_arguments: dict[str, object] = {
        "service": "processFulltextDocument",
        "input_path": str(input_dir),
        "n": workers,
        "force": True,
    }

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=("output", "output_path"),
        value=str(OUTPUT_DIR),
        required=True,
    )

    generated_ids_parameter = set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=("generateIDs", "generate_ids"),
        value=True,
    )

    coordinates_parameter = set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=("tei_coordinates", "teiCoordinates"),
        value=True,
    )

    sentences_parameter = set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=(
            "segment_sentences",
            "segmentSentences",
        ),
        value=True,
    )

    # Header and citation consolidation can call external services.
    # They are unnecessary because article metadata are stored
    # separately in this project.
    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=(
            "consolidate_header",
            "consolidateHeader",
        ),
        value=False,
    )

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=(
            "consolidate_citations",
            "consolidateCitations",
        ),
        value=False,
    )

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=(
            "include_raw_citations",
            "includeRawCitations",
        ),
        value=False,
    )

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=(
            "include_raw_affiliations",
            "includeRawAffiliations",
        ),
        value=False,
    )

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=("markdown_output",),
        value=False,
    )

    set_supported_argument(
        process_arguments,
        process_parameters,
        aliases=("json_output",),
        value=False,
    )

    print("GROBID extraction options:")
    print(
        "- XML IDs: "
        f"{'enabled' if generated_ids_parameter else 'unsupported'}"
    )
    print(
        "- PDF coordinates: "
        f"{'enabled' if coordinates_parameter else 'unsupported'}"
    )
    print(
        "- Sentence segmentation: "
        f"{'enabled' if sentences_parameter else 'unsupported'}"
    )
    print(
        "- Coordinate elements: "
        + ", ".join(GROBID_COORDINATE_ELEMENTS)
    )

    return process_arguments


def process_directory(
    input_dir: Path,
    workers: int,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = create_client()

    print(f"Processing PDFs with GROBID from: {input_dir}")
    print(f"TEI output directory: {OUTPUT_DIR}")

    process_arguments = build_process_arguments(
        client=client,
        input_dir=input_dir,
        workers=workers,
    )

    client.process(**process_arguments)

    print("GROBID processing completed.")


def find_generated_tei(
    article_id: str,
) -> Path | None:
    expected_path = (
        OUTPUT_DIR
        / f"{article_id}.grobid.tei.xml"
    )

    if expected_path.exists():
        return expected_path

    matching_files = sorted(
        OUTPUT_DIR.glob(
            f"{article_id}*.tei.xml"
        )
    )

    if matching_files:
        return matching_files[0]

    return None


def count_coordinated_elements(
    root: ET.Element,
) -> int:
    return sum(
        1
        for element in root.iter()
        if element.attrib.get("coords")
    )


def validate_generated_tei(
    tei_path: Path,
) -> dict[str, int]:
    """
    Verify that the generated TEI contains the requested enrichment.
    """
    try:
        tree = ET.parse(tei_path)
    except ET.ParseError as error:
        raise ValueError(
            f"Invalid TEI XML generated: {tei_path}"
        ) from error

    root = tree.getroot()

    paragraph_count = len(
        root.findall(".//tei:p", NS)
    )

    sentence_count = len(
        root.findall(".//tei:s", NS)
    )

    note_count = len(
        root.findall(".//tei:note", NS)
    )

    figure_count = len(
        root.findall(".//tei:figure", NS)
    )

    coordinated_element_count = (
        count_coordinated_elements(root)
    )

    statistics = {
        "paragraphs": paragraph_count,
        "sentences": sentence_count,
        "notes": note_count,
        "figures": figure_count,
        "coordinated_elements": (
            coordinated_element_count
        ),
    }

    print(f"TEI validation for: {tei_path.name}")
    print(f"- Paragraphs: {paragraph_count}")
    print(f"- Sentences: {sentence_count}")
    print(f"- Notes: {note_count}")
    print(f"- Figures/tables: {figure_count}")
    print(
        "- Elements with PDF coordinates: "
        f"{coordinated_element_count}"
    )

    if coordinated_element_count == 0:
        print(
            "Warning: no PDF coordinates were found. "
            "Check the installed client and GROBID server version."
        )

    if sentence_count == 0:
        print(
            "Warning: no <s> elements were found. "
            "Sentence segmentation may not be enabled "
            "on the GROBID server."
        )

    if paragraph_count == 0:
        print(
            "Warning: no body paragraphs were extracted."
        )

    return statistics


def process_single_pdf(
    pdf_path: Path,
    workers: int,
) -> None:
    article_id = pdf_path.stem

    # The client processes directories, so use a temporary input folder.
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_input_dir = Path(
            temporary_directory
        )

        temporary_pdf_path = (
            temporary_input_dir
            / pdf_path.name
        )

        shutil.copy2(
            pdf_path,
            temporary_pdf_path,
        )

        process_directory(
            input_dir=temporary_input_dir,
            workers=workers,
        )

    generated_tei = find_generated_tei(
        article_id
    )

    if generated_tei is None:
        raise RuntimeError(
            "GROBID did not generate a TEI file "
            f"for: {article_id}"
        )

    print(f"TEI file generated: {generated_tei}")

    validate_generated_tei(
        generated_tei
    )


def process_all_pdfs(
    workers: int,
) -> None:
    pdf_files = find_all_pdfs()

    print(f"PDF files found: {len(pdf_files)}")

    process_directory(
        input_dir=PDF_DIR,
        workers=workers,
    )

    generated_count = 0
    missing_articles: list[str] = []
    invalid_articles: list[str] = []

    for pdf_path in pdf_files:
        article_id = pdf_path.stem
        generated_tei = find_generated_tei(
            article_id
        )

        if generated_tei is None:
            missing_articles.append(
                article_id
            )
            continue

        generated_count += 1

        try:
            validate_generated_tei(
                generated_tei
            )
        except (ValueError, OSError) as error:
            invalid_articles.append(
                f"{article_id}: {error}"
            )

    print(
        f"TEI files generated: "
        f"{generated_count}/{len(pdf_files)}"
    )

    if missing_articles:
        print("Missing TEI files:")

        for article_id in missing_articles:
            print(f"- {article_id}")

    if invalid_articles:
        print("Invalid TEI files:")

        for error_message in invalid_articles:
            print(f"- {error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract scientific article PDFs as enriched "
            "GROBID TEI XML."
        )
    )

    selection_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    selection_group.add_argument(
        "--article",
        type=str,
        help=(
            "Article number or name, "
            "for example: 2 or article_2."
        ),
    )

    selection_group.add_argument(
        "--all",
        action="store_true",
        help="Process all article PDFs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of concurrent GROBID requests. "
            "Use 1 for a local CPU-based server."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error(
            "--workers must be at least 1."
        )

    if args.article:
        pdf_path = find_pdf(
            args.article
        )

        process_single_pdf(
            pdf_path=pdf_path,
            workers=args.workers,
        )
    else:
        process_all_pdfs(
            workers=args.workers,
        )


if __name__ == "__main__":
    main()