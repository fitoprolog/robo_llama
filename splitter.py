import re
import json

def split_entries(text):
    # Split por patrones de encabezados léxicos (todo en mayúsculas, posiblemente con guiones)
    return re.split(r'\n(?=[A-Z][A-Z0-9\- ]+\n)', text)

def extract_entry_data(entry):
    lines = entry.strip().splitlines()
    if not lines:
        return None

    word = lines[0].strip()
    body = "\n".join(lines[1:]).strip()

    # Buscar etimología
    etym_match = re.search(r'Etym: (.+?)(?:\n|Defn:|$)', body, re.DOTALL)
    etymology = etym_match.group(1).strip() if etym_match else None

    # Buscar definición
    defn_match = re.search(r'Defn: (.+?)(?:\n[A-Z][a-z]|$)', body, re.DOTALL)
    definition = defn_match.group(1).strip() if defn_match else body

    # Buscar ejemplos entre comillas o con nombres de autores
    examples = re.findall(r'“([^”]+)”|"(.*?)"|(?:\.\s)?([A-Z][a-z]+\.)', body)
    cleaned_examples = []
    for ex in examples:
        # ex es una tupla con varios grupos, filtramos vacíos
        cleaned = next((e for e in ex if e), None)
        if cleaned:
            cleaned_examples.append(cleaned.strip())

    return {
        "word": word,
        "etymology": etymology,
        "definition": definition,
        "examples": cleaned_examples if cleaned_examples else None
    }

def process_dictionary(raw_text):
    entries = split_entries(raw_text)
    data = []
    for e in entries:
        entry = extract_entry_data(e)
        if entry:
            data.append(entry)
    return data

def save_to_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

