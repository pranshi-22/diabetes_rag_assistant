from pathlib import Path
import re


# ==========================================
# 1. Define folders
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FOLDER = PROJECT_ROOT / "data" / "knowledge_base" / "processed"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "knowledge_base" / "chunks"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Clean extracted text
# ==========================================

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ==========================================
# 3. Create overlapping chunks
# ==========================================

def create_chunks(text, chunk_size=1000, overlap=200):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ==========================================
# 4. Process all text files
# ==========================================

def main():

    text_files = list(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print("No text files found!")
        return

    print(f"Found {len(text_files)} text files.")

    total_chunks = 0

    for text_file in text_files:

        print("\n" + "=" * 60)
        print(f"Processing: {text_file.name}")
        print("=" * 60)

        text = text_file.read_text(encoding="utf-8")

        cleaned_text = clean_text(text)

        chunks = create_chunks(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        output_file = OUTPUT_FOLDER / f"{text_file.stem}_chunks.txt"

        with output_file.open("w", encoding="utf-8") as f:

            for i, chunk in enumerate(chunks, start=1):

                f.write(f"--- CHUNK {i} ---\n")
                f.write(chunk)
                f.write("\n\n")

        total_chunks += len(chunks)

        print(f"Characters before cleaning: {len(text):,}")
        print(f"Characters after cleaning: {len(cleaned_text):,}")
        print(f"Chunks created: {len(chunks):,}")
        print(f"Saved to: {output_file}")

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)
    print(f"Total chunks created: {total_chunks:,}")


# ==========================================
# 5. Run
# ==========================================

if __name__ == "__main__":
    main()