from flask import render_template, request
from app import app
from engine.searcher import search, map_back_to_URL
import json

# Load inverted index and docID mapping into memory
INDEX_FILE = "engine/indexer_json/inverted_index.json"
DOCID_FILE = "engine/indexer_json/doc_id_mapping.json"
inverted_index = json.load(open(INDEX_FILE))
docID_mapping = json.load(open(DOCID_FILE))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search_route():
    query = request.form['query']
    docIDs = search(query, inverted_index)
    urls = map_back_to_URL(docIDs, docID_mapping)
    return render_template('results.html', query=query, results=urls)
