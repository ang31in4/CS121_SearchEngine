import json
import os
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict

## GLOBAL VARIABLES ##
docs_indexed = 0 # number of documents indexed
num_tokens = 0 # number of unique tokens

def tokenize(text):
    """
    Tokenizes the given text into words, preserving alphanumeric sequences and apostrophes.
    Additionally, uses Porter stemming.
    """
    # Initialize the PorterStemmer
    stemmer = PorterStemmer()

    # Tokenize the text into alphanumeric sequences and apostrophes
    text_tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())

    # Apply Porter stemming to each token
    stemmed_tokens = [stemmer.stem(token) for token in text_tokens]

    return stemmed_tokens

def parse_files(path: str):
    global docs_indexed
    inverted_index = defaultdict(list)  # Inverted index (token -> [{docID, freq}])
    doc_id_mapping = {}  # Maps docID -> URL

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

            # Assign a unique docID (zero-indexing)
            docID = docs_indexed
            doc_id_mapping[docID] = url  # Store mapping
            docs_indexed += 1
            print(f'DOCUMENT {docID}: {url}')

            soup = BeautifulSoup(content, 'lxml')
            text_content = soup.get_text()

            tokens = tokenize(text_content)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for token, freq in term_freq.items():
                inverted_index[token].append((docID, freq))

    store_index(inverted_index, doc_id_mapping)

def store_index(inverted_index, doc_id_mapping):
    global num_tokens
    output_index_path = os.path.join(os.getcwd(), 'indexer_json', 'inverted_index.json')
    output_mapping_path = os.path.join(os.getcwd(), 'indexer_json', 'doc_id_mapping.json')

    with open(output_index_path, 'w', encoding='utf-8') as file:
        json.dump(dict(inverted_index), file, separators=(',', ':'))

    with open(output_mapping_path, 'w', encoding='utf-8') as file:
        json.dump(doc_id_mapping, file, separators=(',', ':'))

    num_tokens = len(inverted_index)

    print(f'Stored inverted index: {output_index_path}')
    print(f'Stored docID mapping: {output_mapping_path}')


def build_report() -> None:
    global num_tokens, docs_indexed
    docID_file_path = os.path.join(os.getcwd(), 'indexer_json','doc_id_mapping.json')
    inverted_index_path = os.path.join(os.getcwd(), 'indexer_json', 'inverted_index.json')
    report_file_path = os.path.join(os.getcwd(), 'report.txt')

    # Convert file size of files in indexer_json to KB
    docID_bytes = os.stat(docID_file_path).st_size
    inverted_index_bytes = os.stat(inverted_index_path).st_size
    json_kb = (docID_bytes + inverted_index_bytes) / 1024

    # Write report details
    with open(report_file_path, 'w', encoding = 'utf-8') as report_file:
        report_file.write(f'DOCUMENTS INDEXED: {docs_indexed}\n')
        report_file.write(f'UNIQUE TOKENS: {num_tokens}\n')
        report_file.write(f'TOTAL SIZE (IN KB): {json_kb:.2f} KB\n')

if __name__ == '__main__':
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    directory_path = os.path.join(parent_dir, 'DEV')

    parse_files(directory_path)
    build_report()