from flask import render_template, request
from app import app
from engine.searcher import search, map_back_to_URL
from engine.summaries import generate_summaries
import json

# Load inverted index and docID mapping into memory
#INDEX_FILE = "engine/indexer_json/inverted_index.json"
DOCID_FILE = "engine/indexer_json/merged_docIDs.json"
#inverted_index = json.load(open(INDEX_FILE))
docID_mapping = json.load(open(DOCID_FILE))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search_route():
    query = request.form['query']
    docIDs = search(query)
    urls = map_back_to_URL(docIDs, docID_mapping)
    summaries = generate_summaries(urls)
    outputs = zip(urls, summaries)

    return render_template('results.html', query=query, results=urls, outputs=outputs)
