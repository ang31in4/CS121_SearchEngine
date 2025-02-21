import json
import os
import re
from bs4 import BeautifulSoup
from collections import defaultdict

def tokenize(text):
    """
        Tokenizes the given text into words, preserving alphanumeric sequences and apostrophes.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list: A list of tokens extracted from the text.
    """
    # Split by non-word characters and convert to lowercase
    text_tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return text_tokens


def parse_files(path: str) -> int:
    # Initialize the inverted index and documents indexed
    inverted_index = defaultdict(list)
    documents_indexed = 0

    # Parse through folders of DEV
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                url = data.get('url', '')
                content = data.get('content', '')

                documents_indexed += 1
                print(f'DOCUMENT {documents_indexed}: {url}')

                soup = BeautifulSoup(content, 'xml')
                text_content = soup.get_text()

                # Tokenize the content
                tokens = tokenize(text_content)

                # Calculate term frequency
                term_freq = defaultdict(int)
                for token in tokens:
                    term_freq[token] += 1

                # Update the inverted index with the token frequencies and URLs
                for token, freq in term_freq.items():
                    inverted_index[token].append({
                        'url': url,
                        'term_frequency': freq
                    })

    store_index(inverted_index)
    return documents_indexed

def store_index(inverted_index):
    # Convert defaultdict to a regular dict
    inverted_index = dict(inverted_index)

    output_file_path = os.path.join(parent_dir, 'indexer', 'inverted_index.json')

    # Write the inverted index to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(inverted_index, file, indent=2)

    # Print total number of tokens
    print(f'NUMBER OF TOKENS: {len(inverted_index)}')


if __name__ == '__main__':
    # Get path to DEV folder
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    directory_path = os.path.join(parent_dir, 'DEV')

    # Parse through documents
    num_indexed = parse_files(directory_path)
    print(f'NUMBER OF DOCUMENTS: {num_indexed}')
