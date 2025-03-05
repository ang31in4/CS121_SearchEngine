import json
import os
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict


## GLOBAL VARIABLES ##
docs_indexed = 0 # number of documents indexed
num_tokens = 0 # number of unique tokens
json_batch = 0 # number of json batches for partial indexing


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


def parse_files(path: str) -> None:
    """
    Parses through DEV folder and indexes all documents.

    :param path: path to DEV folder
    """
    global docs_indexed
    global json_batch
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

            # Dump into new json file every 10000 documents
            if docs_indexed % 1000 == 0:
                store_index(inverted_index, doc_id_mapping)
                json_batch += 1
                # Clear from memory
                inverted_index.clear()
                doc_id_mapping.clear()

    store_index(inverted_index, doc_id_mapping) # dump remaining documents

    merge_docIDs() # merge docIDs
    merge_index() # merge inverted_indexes


def store_index(inverted_index: defaultdict, doc_id_mapping: dict) -> None:
    """
    Dumps inverted_index and doc_id_mapping into JSON files.

    :param inverted_index:
    :param doc_id_mapping:
    """
    global num_tokens
    global json_batch

    # Add the batch number to the filenames
    output_index_path = os.path.join(os.getcwd(), 'indexer_json',
                                     f'inverted_index_{json_batch}.json')
    output_mapping_path = os.path.join(os.getcwd(), 'indexer_json',
                                       f'doc_id_mapping_{json_batch}.json')

    # Store the inverted index
    with open(output_index_path, 'w', encoding = 'utf-8') as file:
        json.dump(dict(inverted_index), file, separators = (',', ':'))

    # Store the doc ID mapping
    with open(output_mapping_path, 'w', encoding = 'utf-8') as file:
        json.dump(doc_id_mapping, file, separators = (',', ':'))

    print(f'Batch {json_batch} of inverted index stored in: {output_index_path}')
    print(f'Batch {json_batch} of docID mapping stored in: {output_mapping_path}')


def merge_docIDs() -> None:
    """
    Merges all batches of docID mappings stored into one JSON file.
    """
    global json_batch
    # Open the output file in write mode
    merged_docIDs = os.path.join(os.getcwd(), 'indexer_json', 'merged_docIDs.json')

    # Merge all batches of docIDs
    with open(merged_docIDs, 'w', encoding='utf-8') as output_file:
        first_batch = True

        # Iterate over batch files
        i = 0
        while i <= json_batch:
            batch_file = os.path.join(os.getcwd(), 'indexer_json', f'doc_id_mapping_{i}.json')
            with open(batch_file, 'r', encoding='utf-8') as file:
                data = json.load(file)  # Read the current batch

                if first_batch:
                    # Write the start of the JSON structure
                    output_file.write('{')  # Start of the JSON object
                    first_batch = False
                else:
                    # Add a comma between merged batches
                    output_file.write(',')

                # Write the inner key-value pairs only, removing unnecessary spaces
                output_file.write(json.dumps(data, separators=(',', ':'))[1:-1])

            # Delete file
            os.remove(batch_file)
            i += 1

        # Write the closing brace to end the JSON structure
        output_file.write('}')

    print(f'Merged docIDs has been stored in: {merged_docIDs}')


def merge_index() -> None:
    """
    Merges all batches of inverted index stored into one JSON file.
    """
    global json_batch
    global num_tokens
    # Open the output file in write mode
    merged_inverted_index = os.path.join(os.getcwd(), 'indexer_json', 'merged_inverted_index.json')

    # Initialize an empty dictionary to hold the merged data
    merged_data = {}

    # Open the output file in write mode
    with open(merged_inverted_index, 'w', encoding = 'utf-8') as output_file:

        # Iterate over batch files and merge them incrementally
        i = 0
        while i <= json_batch:
            batch_file = os.path.join(os.getcwd(), 'indexer_json', f'inverted_index_{i}.json')
            with open(batch_file, 'r', encoding = 'utf-8') as file:
                data = json.load(file)  # Read the current batch

                # For each key in the batch file, merge the data with the output
                for key, value in data.items():
                    # If the key is already in the merged data, merge the lists
                    if key in merged_data:
                        merged_data[key].extend(value)
                    else:
                        # If the key doesn't exist, add the new key and value
                        merged_data[key] = value

            # Delete the batch file after processing
            os.remove(batch_file)
            i += 1

        # After processing all batches, write the closing brace to end the JSON structure
        json.dump(merged_data, output_file, separators = (',', ':'))
        output_file.write('}')

    num_tokens = len(merged_data)

    print(f'Merged inverted index has been stored in: {merged_inverted_index}')


def build_report() -> None:
    global num_tokens, docs_indexed
    docID_file_path = os.path.join(os.getcwd(), 'indexer_json','merged_docIDs.json')
    inverted_index_path = os.path.join(os.getcwd(), 'indexer_json', 'merged_inverted_index.json')
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