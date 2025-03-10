import json
import string
import os
import re
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from collections import defaultdict
import engine.simhash

## GLOBAL VARIABLES ##
docs_indexed = 0  # number of documents indexed
num_tokens = 0  # number of unique tokens
json_batch = 0  # number of json batches for partial indexing


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
    global docs_indexed, json_batch
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

            print(f'DOCUMENT {docID}: {url}')

            soup = BeautifulSoup(content, 'lxml')
            text_content = soup.get_text()

            tokens = tokenize(text_content)

            if engine.simhash.is_same_content(tokens):
                continue

            docs_indexed += 1

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for token, freq in term_freq.items():
                inverted_index[token].append((docID, freq))  # add a third parameter to the tuple which is tf-idf

            # Dump into new json file every 10000 documents
            if docs_indexed % 10000 == 0:
                store_index(inverted_index, doc_id_mapping)
                json_batch += 1
                # Clear from memory
                inverted_index.clear()
                doc_id_mapping.clear()

    store_index(inverted_index, doc_id_mapping)  # dump remaining documents

    merge_docIDs()  # merge docIDs
    sort_index()  # sort inverted_indexes


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
    with open(output_index_path, 'w', encoding='utf-8') as file:
        json.dump(dict(inverted_index), file, separators=(',', ':'))

    # Store the doc ID mapping
    with open(output_mapping_path, 'w', encoding='utf-8') as file:
        json.dump(doc_id_mapping, file, separators=(',', ':'))

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


def sort_index() -> None:
    """
    Merges all batches of inverted index stored into separate JSON files
    sorted by the first letter of each term (e.g., a_inverted_index.json, b_inverted_index.json).
    """
    global json_batch, num_tokens

    # Initialize output directory and files
    output_dir = os.path.join(os.getcwd(), 'indexer_json')

    # Create a defaultdict to store merged results for each letter-based index
    letter_based_index = defaultdict(lambda: defaultdict(list))

    i = 0
    while i <= json_batch:
        batch_file = os.path.join(output_dir, f'inverted_index_{i}.json')
        with open(batch_file, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Read the current batch

            # For each key in the batch file, merge the data with the output
            for key, value in data.items():
                # Determine the first letter of the term (case insensitive)
                first_letter = key[0].lower()
                if first_letter.isdigit():
                    first_letter = 'numbers'
                elif not first_letter.isalpha():
                    first_letter = 'special'

                # Append the value to the letter-based index
                letter_based_index[first_letter][key].extend(value)

        # Delete the batch file after processing
        os.remove(batch_file)

        # Write the merged data to disk
        for first_letter, sorted_index in letter_based_index.items():
            sorted_file_path = os.path.join(output_dir, f'{first_letter}_inverted_index.jsonl')

            # If the sorted file exists, load it, otherwise create a new one
            existing_index = {}
            if os.path.exists(sorted_file_path):
                with open(sorted_file_path, 'r', encoding='utf-8') as sorted_file:
                    for line in sorted_file:
                        try:
                            entry = json.loads(line.strip())
                            term, postings = next(iter(entry.items()))
                            existing_index[term] = postings
                        except json.JSONDecodeError:
                            continue  # Skip corrupted lines

            # Merge the new data with the existing index
            for key, value in sorted_index.items():
                if key in existing_index:
                    existing_index[key].extend(value)
                else:
                    existing_index[key] = value
                    num_tokens += 1

            # Write the merged data back to the sorted file
            with open(sorted_file_path, 'w', encoding='utf-8') as sorted_file:
                for term, postings in sorted_index.items():
                    json.dump({term: postings}, sorted_file)
                    sorted_file.write('\n')  # Newline after each JSON object

        letter_based_index.clear()  # Clear memory
        i += 1

    print(f'Sorted inverted index has been stored in letter-based index files under: {output_dir}')


def build_index_of_index():
    '''
    Builds the index of term locations in the partial indexes
    '''

    index_dir = "indexer_json"  # Directory containing all JSONL index files
    offset_index_file = "index_offsets.json"
    offsets = {}
    # Iterate over all index files
    for filename in os.listdir(index_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(index_dir, filename)

            # Open each JSONL file and record term offsets
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()  # Get the starting byte position
                    line = f.readline()
                    if not line:
                        break  # Stop when EOF
                    try:
                        data = json.loads(line)  # Load the JSON line
                        term = next(iter(data))  # Extract the term (first key in dictionary)
                        offsets[term] = offset  # Store term and offset
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

    # Save the offset index to a JSON file
    with open(os.path.join(index_dir, offset_index_file), 'w', encoding='utf-8') as f:
        json.dump(offsets, f, separators=(',', ':'))

    print(f"Offset index stored in {offset_index_file}")


def build_report() -> None:
    """
    Build report that has index statistics (number documents
    indexed, number of tokens, total disk space (in KB).
    """
    global num_tokens, docs_indexed
    sorted_index_letters = list(string.ascii_lowercase) + ['numbers',
                                                           'special']  # Make a list of inverted_index json names
    docID_file_path = os.path.join(os.getcwd(), 'indexer_json', 'merged_docIDs.json')
    report_file_path = os.path.join(os.getcwd(), 'report.txt')

    # Convert file size of files in indexer_json to KB
    docID_bytes = os.stat(docID_file_path).st_size
    inverted_index_bytes = 0
    for letter in sorted_index_letters:
        file_path = os.path.join(os.getcwd(), 'indexer_json', f'{letter}_inverted_index.jsonl')

        # Check if the file exists, and add its size if it does
        if os.path.exists(file_path):
            inverted_index_bytes += os.stat(file_path).st_size

    json_kb = (docID_bytes + inverted_index_bytes) / 1024

    # Write report details
    with open(report_file_path, 'w', encoding='utf-8') as report_file:
        report_file.write(f'DOCUMENTS INDEXED: {docs_indexed}\n')
        report_file.write(f'UNIQUE TOKENS: {num_tokens}\n')
        report_file.write(f'TOTAL SIZE (IN KB): {json_kb:.2f} KB\n')


if __name__ == '__main__':
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    directory_path = os.path.join(parent_dir, 'DEV')

    parse_files(directory_path)
    build_report()
    build_index_of_index()