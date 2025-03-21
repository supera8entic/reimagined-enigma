import streamlit as st
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image
import os
import tempfile
import chromadb
from chromadb.utils import embedding_functions
import groqai

# Initialize ChromaDB with embedding function
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.create_collection(name="rbi_compliance", embedding_function=embedding_fn)

# Initialize SmolDocling model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Check if the URL points to an image or document
        content_type = response.headers.get('Content-Type', '')
        
        if 'image' in content_type:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            # Process image with SmolDocling
            image = Image.open(tmp_file_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
            generated_ids = model.generate(**inputs, max_new_tokens=8192)
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]
            doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
            
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
            doc = DoclingDocument(name="Document")
            doc.load_from_doctags(doctags_doc)
            text = doc.export_to_markdown()
            
            os.remove(tmp_file_path)
            return text
        
        elif 'pdf' in content_type or 'html' in content_type:
            # For HTML content, extract text with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            return text
        
        else:
            st.write(f"Content type {content_type} not supported")
            return None
    
    except Exception as e:
        st.write(f"Error extracting text from URL: {e}")
        return None

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def save_chunks_to_db(url, chunks):
    try:
        metadata = [{"source": url, "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, metadatas=metadata)
        st.write("Chunks saved to database")
    except Exception as e:
        st.write(f"Error saving chunks to database: {e}")

def query_database(query_text, k=3):
    try:
        results = collection.query(query_texts=[query_text], n_results=k)
        return results['documents'][0]
    except Exception as e:
        st.write(f"Error querying database: {e}")
        return []

def generate_response_with_groq(query_text):
    try:
        response = groqai.complete(prompt=query_text, max_tokens=200)
        return response
    except Exception as e:
        st.write(f"Error generating response with Groq AI: {e}")
        return "I'm sorry, I'm unable to generate a response at this time."

st.title("RBI Compliance Advisory Tool")

# URL input
url = st.text_input("Enter URL to process:")

if st.button("Process URL"):
    if url:
        st.write(f"Processing URL: {url}")
        text = extract_text_from_url(url)
        if text:
            chunks = chunk_text(text)
            save_chunks_to_db(url, chunks)
    else:
        st.write("Please enter a URL")

# Query input
query = st.text_input("Enter your compliance question:")

if st.button("Ask Question"):
    if query:
        st.write(f"Searching for: {query}")
        results = query_database(query)
        
        if results:
            st.write("Found relevant information:")
            for result in results:
                st.write(result)
        else:
            st.write("No relevant information found. Generating response with Groq AI...")
            response = generate_response_with_groq(query)
            st.write(response)
    else:
        st.write("Please enter a question")
