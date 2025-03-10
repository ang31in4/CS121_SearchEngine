Overview
This application provides a search interface. Users can index documents and then search for any word or phrase. The top relevant results are displayed, and clicking on any result opens a new window.

Prerequisites
Python 3.0 or later installed
nltk
lxml
beautifulsoup4
flask
python-dotenv
langchain-community
transformers
torch

Setup and Usage

Index the Data
- make a directory under engine called indexer_json
- go into the engine directory
cd engine
-Run the indexer.py file (the script that builds or updates your search index).
python indexer.py
-This step processes your data and prepares it for searching.

Running it locally:
- navigate to the engine directory if not already there
- run the searcher.py file
	python searcher.py
- enter query to search for
- q to quit

Launch the Search Application
-From the command line, navigate to the directory containing your Flask application (for instance, the directory with app.py, or wherever your Flask entry point is located).
-Run the Flask server with the command:
flask run --host=0.0.0.0
-This will start the Flask development server on your machine, listening on all network interfaces.

Performing a Search
-Open your web browser and go to the address displayed in the terminal (e.g., http://127.0.0.1:5000/ by default).
-Type your search word or phrase into the search bar.
-Click the Search button (or press Enter).
-The top relevant results will be displayed. Click on any result to open it in a new window.

How It Works
Index Creation: The indexer.py script gathers data from your source files, processes text, and stores it in a searchable index.
Searcher: Uses the index_offset files to locate the location of the terms from the queries. Navigates to corresponding partial index, retrives relevant postings, uses the tf-idf score to rank documents and displays the top 5 results.
Server: Flask serves as the front-end interface. When you start the server, Flask handles user requests to search the indexed data.

Troubleshooting
-Make sure the indexing process completes successfully. If the index is not built, you will not see any results.
-If indexing is not working, make sure there is a directory in the engine directory named ‘indexer_json’


