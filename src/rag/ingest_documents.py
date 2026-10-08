import os
import re
from pypdf import PdfReader


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge_base",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "knowledge_base",
    "processed"
)


os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_pdf_text(text):
    """
    Clean common PDF extraction artifacts while preserving
    meaningful medical content.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Remove page-number-only lines
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    # Remove common navigation phrases
    navigation_patterns = [
        r"(?im)^\s*return to overview page\s*$",
        r"(?im)^\s*return to top\s*$",
        r"(?im)^\s*back to top\s*$",
        r"(?im)^\s*previous page\s*$",
        r"(?im)^\s*next page\s*$",
    ]

    for pattern in navigation_patterns:
        text = re.sub(pattern, "", text)

    # Remove repeated whitespace
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Collapse excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def extract_pdf(pdf_path):
    """
    Extract text from all pages of a PDF.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


def save_processed_text(filename, text):

    output_name = os.path.splitext(filename)[0] + ".txt"

    output_path = os.path.join(
        PROCESSED_DIR,
        output_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    return output_path


def main():

    pdf_files = [
        filename
        for filename in os.listdir(RAW_DIR)
        if filename.lower().endswith(".pdf")
    ]

    print("=" * 60)
    print("DIABETES KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    for filename in pdf_files:

        pdf_path = os.path.join(
            RAW_DIR,
            filename
        )

        print(f"\nProcessing: {filename}")

        raw_text = extract_pdf(pdf_path)

        print(
            f"Extracted characters: {len(raw_text):,}"
        )

        cleaned_text = clean_pdf_text(
            raw_text
        )

        print(
            f"Cleaned characters: {len(cleaned_text):,}"
        )

        output_path = save_processed_text(
            filename,
            cleaned_text
        )

        print(
            f"Saved: {output_path}"
        )

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()