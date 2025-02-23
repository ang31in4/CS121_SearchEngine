import json
import os
import re
from bs4 import BeautifulSoup
from collections import defaultdict

def tokenize(text):
    """
    Tokenizes the given text into words, preserving alphanumeric sequences and apostrophes.
    """
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def parse_files(path: str) -> int:
    inverted_index = defaultdict(list)  # Inverted index (token -> [{docID, freq}])
    doc_id_mapping = {}  # Maps docID -> URL
    documents_indexed = 0

    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Skipping {file_path}: {e}")
                continue

            url = data.get('url', '')
            content = data.get('content', '')

            if not url or not content:
                continue  # Skip invalid documents

            # Assign a unique docID
            docID = documents_indexed
            doc_id_mapping[docID] = url  # Store mapping
            documents_indexed += 1
            print(f'DOCUMENT {docID}: {url}')

            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()

            tokens = tokenize(text_content)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for token, freq in term_freq.items():
                inverted_index[token].append((docID, freq))

    store_index(inverted_index, doc_id_mapping)
    return documents_indexed

def store_index(inverted_index, doc_id_mapping):
    output_index_path = os.path.join(os.getcwd(), 'inverted_index.json')
    output_mapping_path = os.path.join(os.getcwd(), 'doc_id_mapping.json')

    with open(output_index_path, 'w', encoding='utf-8') as file:
        json.dump(dict(inverted_index), file, separators=(',', ':'))

    with open(output_mapping_path, 'w', encoding='utf-8') as file:
        json.dump(doc_id_mapping, file, separators=(',', ':'))

    print(f'NUMBER OF TOKENS: {len(inverted_index)}')
    print(f'Stored inverted index: {output_index_path}')
    print(f'Stored docID mapping: {output_mapping_path}')

if __name__ == '__main__':
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    directory_path = os.path.join(parent_dir, 'DEV')

    num_indexed = parse_files(directory_path)
    print(f'NUMBER OF DOCUMENTS: {num_indexed}')