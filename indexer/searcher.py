import json
import indexer

def load_index(index_file):
    """Load the entire inverted index into memory."""
    with open(index_file, 'r') as f:
        return json.load(f)

def search(query, inverted_index):
    """Perform a logical AND search on the loaded inverted index."""
    query_terms = indexer.tokenize(query)  # Convert query to set
    doc_sets = []  # Store sets of docIDs for each term
    doc_freq_map = {}  # Stores {docID: total frequency}


    for term in query_terms:
        if term in inverted_index:
            postings = inverted_index[term]  # List of [docID, freq]
            doc_ids = {doc[0] for doc in postings}
            doc_sets.append(doc_ids)

            # Accumulate frequency scores
            for doc_id, freq in postings:
                doc_freq_map[doc_id] = doc_freq_map.get(doc_id, 0) + freq

    if not doc_sets:
        return []  # No matching terms

    doc_sets.sort(key=len)  # sort the doc_sets, so we can process the shortest token postings first

    # Perform logical AND by intersecting incrementally
    result_docs = doc_sets[0]
    for doc_set in doc_sets[1:]:
        result_docs &= doc_set  # Efficient in-place intersection

        # Early exit: If the intersection is empty, no need to continue
        if not result_docs:
            return set()

    sorted_results = sorted(result_docs, key=lambda doc: doc_freq_map[doc], reverse=True)
    return sorted_results[:5]  # Return top 5 results


def map_back_to_URL(docID, docID_mapping):
    urls = []
    for id in docID:
        urls.append(docID_mapping.get(str(id)))

    return urls


if __name__ == "__main__":
    index_file = "indexer_json/inverted_index.json"
    docID_file = "indexer_json/doc_id_mapping.json"

    inverted_index = load_index(index_file)  # Load once into memory
    docID_file = load_index(docID_file)

    while True:
        user_query = input("Enter your search query (or 'q' to quit): ").strip()
        if user_query.lower() == 'q':
            break

        docIDs = search(user_query, inverted_index)
        urls = map_back_to_URL(docIDs, docID_file)

        print("Matching documents:", urls if urls else "No matching documents found.")
