import re
import json
from typing import List, Dict


def split_by_sections(text: str) -> List[Dict]:
    pattern = r'(?m)^(?P<section>\d+(\.\d+)*)\s+(?P<title>.+)'
    matches = list(re.finditer(pattern, text))

    sections = []

    for i, match in enumerate(matches):
        section_number = match.group("section")
        title = match.group("title")

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        sections.append({
            "section": section_number,
            "title": title,
            "content": content
        })

    return sections


def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(text: str,
                chunk_size=675,
                overlap=100) -> List[Dict]:

    sections = split_by_sections(text)
    final_chunks = []

    for section in sections:

        section_header = f"Section {section['section']}: {section['title']}\n\n"
        full_text = section_header + section["content"]

        subchunks = chunk_text(full_text, chunk_size, overlap)

        for i, chunk in enumerate(subchunks):
            final_chunks.append({
                "text": chunk,
                "metadata": {
                    "section": section["section"],
                    "title": section["title"],
                    "chunk_index": i
                }
            })

    return final_chunks


def save_chunks_to_json(chunks: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(chunks, json_file, indent=4, ensure_ascii=False)


def load_chunks_from_json(input_path: str) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


if __name__ == "__main__":


    input_path = "C://Users//ASUS//Desktop//HallucinationRadar//datasetPart//1.MANAGEMENT.txt"


    output_path = "C://Users//ASUS//Desktop//HallucinationRadar//datasetPart//1.MANAGEMENT_chunks.json"


    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=100
    )

    save_chunks_to_json(chunks, output_path)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")

    if len(chunks) > 0:
        print("\nExample chunk:\n")
        print(chunks[0]["text"])

        print("\nMetadata:\n")
        print(chunks[0]["metadata"])

    loaded_chunks = load_chunks_from_json(output_path)
    print(f"\nLoaded chunks from JSON: {len(loaded_chunks)}")