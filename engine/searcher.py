import json
import os
from indexer import tokenize
import time
import math
from collections import Counter

INDEX_DIR = "indexer_json"

def get_index_file(term):
    """Determine the appropriate index file for a given term."""
    first_char = term[0].lower()
    if 'a' <= first_char <= 'z':  # Alphabet files
        return os.path.join(INDEX_DIR, f'{first_char}_inverted_index.jsonl')
    elif '0' <= first_char <= '9':  # Numbers file
        return os.path.join(INDEX_DIR, 'numbers_inverted_index.jsonl')
    else:  # Special characters file
        return os.path.join(INDEX_DIR, 'special_inverted_index.jsonl')


def load_partial_index(terms, index_offset):
    """Load only the necessary postings from the index using index offsets."""
    partial_index = {}

    for term in terms:
        index_file = get_index_file(term)

        if term in index_offset and os.path.exists(index_file):
            offset = index_offset[term]
            with open(index_file, 'r', encoding='utf-8') as f:
                f.seek(offset)  # Jump to the stored position
                line = f.readline().strip()
                try:
                    entry = json.loads(line)
                    partial_index[term] = entry.get(term, [])
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines

    return partial_index

def cosine_similarity(q_vec, d_vec):
    """Compute cosine similarity between two vectors represented as dictionaries."""
    dot_product = sum(q_vec.get(term, 0) * d_vec.get(term, 0) for term in q_vec)
    norm_q = math.sqrt(sum(val ** 2 for val in q_vec.values()))
    norm_d = math.sqrt(sum(val ** 2 for val in d_vec.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot_product / (norm_q * norm_d)

def search(query, total_docs, index_offset):
    """Perform a logical AND search on the dynamically loaded inverted index."""
    query_terms = tokenize(query)
    print(query_terms)
    if not query_terms:
        return []

    inverted_index = load_partial_index(query_terms, index_offset)
    if not inverted_index:
        return []

    # Build the query vector using tf-idf weights.
    query_tf = Counter(query_terms)
    query_vector = {}
    for term, tf in query_tf.items():
        if term in inverted_index:
            df = len(inverted_index[term])
            idf = math.log(total_docs / df)
            query_vector[term] = (1 + math.log(tf)) * idf

    # Accumulate document scores based on the document's tf-idf for query terms.
    doc_vectors = {}  # Mapping: docID -> { term: weight }
    for term in query_vector:
        if term in inverted_index:
            postings = inverted_index[term]
            df = len(postings)
            idf = math.log(total_docs / df)
            for posting in postings:
                docID, freq = posting
                if docID not in doc_vectors:
                    doc_vectors[docID] = {}
                # If a term appears multiple times in a doc, we use its tf value
                doc_vectors[docID][term] = (1+math.log(freq)) * idf

    scores = {}
    for docID, d_vec in doc_vectors.items():
        # if not all(term in d_vec for term in query_vector): continue
        scores[docID] = cosine_similarity(query_vector, d_vec)

    # Rank documents by score and return top 5.
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [docID for docID, score in ranked_docs[:5]]
    return top_docs


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
    index_offset_path = os.path.join(INDEX_DIR, "index_offsets.json")

    report_file = "search_report.txt"

    # Load docID to URL mapping once
    with open(docID_file_path, 'r', encoding='utf-8') as f:
        docID_mapping = json.load(f)

    # Load index offsets for efficient lookup
    with open(index_offset_path, 'r', encoding='utf-8') as f:
        index_offset = json.load(f)

    while True:
        user_query = input("Enter your search query (or 'q' to quit): ").strip()
        if user_query.lower() == 'q':
            break

        start_time = time.time()

        docIDs = search(user_query, len(docID_mapping), index_offset)
        urls = map_back_to_URL(docIDs, docID_mapping)

        search_time = (time.time() - start_time) * 1000

        write_report(user_query, urls, report_file)

        print("Matching documents:", urls if urls else "No matching documents found.")
        print("Search time =", search_time)
