import json
import os
from indexer import tokenize
import time

INDEX_DIR = "indexer_json"

def get_index_file(term):
    """Determine the appropriate index file for a given term."""
    first_char = term[0].lower()
    if 'a' <= first_char <= 'z':  # Alphabet files
        return os.path.join(INDEX_DIR, f'{first_char}_inverted_index.json')
    elif '0' <= first_char <= '9':  # Numbers file
        return os.path.join(INDEX_DIR, 'numbers_inverted_index.json')
    else:  # Special characters file
        return os.path.join(INDEX_DIR, 'special_inverted_index.json')

def load_partial_index(terms):
    """Load only the necessary portions of the index from relevant files."""
    partial_index = {}

    # Determine unique index files needed
    index_files_needed = {}
    for term in terms:
        index_file = get_index_file(term)
        if index_file not in index_files_needed:
            index_files_needed[index_file] = []
        index_files_needed[index_file].append(term)

    # Load only relevant terms from each index file
    for index_file, terms_in_file in index_files_needed.items():
        if os.path.exists(index_file):  # Ensure the file exists
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                for term in terms_in_file:
                    if term in index_data:
                        partial_index[term] = index_data[term]

    return partial_index

def search(query):
    """Perform a logical AND search on the dynamically loaded inverted index."""
    query_terms = tokenize(query)
    if not query_terms:
        return []

    inverted_index = load_partial_index(query_terms)
    if not inverted_index:
        return []

    doc_sets = []
    doc_freq_map = {}

    for term in query_terms:
        if term in inverted_index:
            postings = inverted_index[term]  # List of [docID, freq]
            doc_ids = {doc[0] for doc in postings}
            doc_sets.append(doc_ids)

            # Accumulate frequency scores
            for doc_id, freq in postings:
                doc_freq_map[doc_id] = doc_freq_map.get(doc_id, 0) + freq

    if not doc_sets:
        return []

    doc_sets.sort(key=len)  # Sort postings lists by size to optimize intersection
    result_docs = doc_sets[0]

    for doc_set in doc_sets[1:]:
        result_docs &= doc_set  # In-place intersection
        if not result_docs:  # Early exit
            return []

    sorted_results = sorted(result_docs, key=lambda doc: doc_freq_map[doc], reverse=True)
    return sorted_results[:5]  # Return top 5 results

def map_back_to_URL(docID, docID_mapping):
    urls = []
    for id in docID:
        urls.append(docID_mapping.get(str(id)))

    return urls

def write_report(query, urls, report_file_path):
    """Write search results to a report file."""
    with open(report_file_path, 'a', encoding='utf-8') as report_file:
        report_file.write(f'Query: {query}\n')
        for url in urls:
            report_file.write(f'{url}\n')
        report_file.write('\n')

if __name__ == "__main__":
    docID_file_path = os.path.join(INDEX_DIR, "merged_docIDs.json")
    report_file = "search_report.txt"

    # Load docID to URL mapping once
    with open(docID_file_path, 'r', encoding='utf-8') as f:
        docID_mapping = json.load(f)

    while True:
        user_query = input("Enter your search query (or 'q' to quit): ").strip()
        if user_query.lower() == 'q':
            break

        start_time = time.time()

        docIDs = search(user_query)
        urls = map_back_to_URL(docIDs, docID_mapping)

        search_time = (time.time() - start_time) * 1000

        write_report(user_query, urls, report_file)

        print("Matching documents:", urls if urls else "No matching documents found.")
        print("Search time =", search_time)