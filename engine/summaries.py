from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.document import Document
from transformers import pipeline

# Load local summarization model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

def generate_document(url):
    """Fetch and return a trimmed langchain Document from a given URL."""
    try:
        loader = WebBaseLoader([url])
        elements = loader.load()
        full_content = " ".join([e.page_content for e in elements])
        trimmed_content = full_content[:1000]  # Limit text to 1000 characters
        return Document(page_content=trimmed_content, metadata={"source": url})
    except Exception as e:
        print(f"Error loading document from {url}: {e}")
        return Document(page_content="", metadata={"source": url})

def summarize_text(text, max_length=130, min_length=30):
    """Summarize text using a local model."""
    try:
        summary = summarizer(text[:1000], max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text'].strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "Failed to generate summary."

def generate_summaries(urls):
    """Fetch and summarize text for each URL."""
    summaries = []
    for url in urls:
        doc = generate_document(url)
        summary = summarize_text(doc.page_content)
        summaries.append(summary)
    return summaries

if __name__ == '__main__':
    urls = [input('Enter URL: ')]
    print(generate_summaries(urls))
