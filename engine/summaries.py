from langchain_community.document_loaders import WebBaseLoader
from langchain_community.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import re
import torch
import asyncio

# Load a smaller, faster quantized summarization model
device = 0 if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6').to(device)

def smart_trim(text, max_chars=1000):
    """Trim text to the nearest sentence without exceeding max_chars."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    trimmed = ""
    for sentence in sentences:
        if len(trimmed) + len(sentence) > max_chars:
            break
        trimmed += sentence + " "
    return trimmed.strip()

def generate_document(url):
    """Fetch and return a trimmed langchain Document from a given URL."""
    try:
        loader = WebBaseLoader([url])
        elements = loader.load()
        full_content = " ".join([e.page_content for e in elements])
        trimmed_content = smart_trim(full_content)
        return Document(page_content=trimmed_content, metadata={"source": url})
    except Exception as e:
        print(f"Error loading document from {url}: {e}")
        return Document(page_content="", metadata={"source": url})

def summarize_texts(texts, max_length=150, min_length=50):
    """Summarize a batch of texts using a quantized model with improved quality."""
    try:
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
        summaries = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True
        )
        return [tokenizer.decode(summary, skip_special_tokens=True).strip() for summary in summaries]
    except Exception as e:
        print(f"Error summarizing texts: {e}")
        return ["Failed to generate summary."] * len(texts)

async def generate_document_async(url):
    return await asyncio.to_thread(generate_document, url)

async def generate_summaries(urls):
    """Fetch and summarize text for each URL in parallel."""
    docs = await asyncio.gather(*(generate_document_async(url) for url in urls))
    texts = [doc.page_content for doc in docs]
    return summarize_texts(texts)

if __name__ == '__main__':
    urls = [input('Enter URL: ')]

    start_time = time.time()

    summaries = asyncio.run(generate_summaries(urls))
    print(summaries)

    search_time = (time.time() - start_time) * 1000

    print("Search time =", search_time)
