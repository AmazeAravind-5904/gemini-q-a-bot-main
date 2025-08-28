!pip install google-generativeai PyMuPDF sentence-transformers scikit-learn faiss-cpu

import os
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List
import numpy as np
import textwrap
import nltk
import faiss
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]

def extract_text(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

document_text = extract_text(filename)
print("Document loaded successfully.")

def split_text(text, max_tokens=300, chunk_size=300, chunk_overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    curr_chunk = []
    curr_length = 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if curr_length + token_count > max_tokens:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = []
            curr_length = 0
        curr_chunk.append(sentence)
        curr_length += token_count

    if curr_chunk:
        chunks.append(" ".join(curr_chunk))

    return chunks

chunks = split_text(document_text)
print(f"Total chunks created: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")  # Required for FAISS
print("Embedding complete.")

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
print(f"FAISS index created with {faiss_index.ntotal} vectors.")

gen_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def ask_gemini(question):
    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Use the following document information to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    response = gen_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.2))
    return str(response.text).strip()

sample_questions = ["Can you explain the Cancellation of the Policy?"]

for question in sample_questions:
    print("Question:", question)
    answer = ask_gemini(question)
    print("Answer:\n", textwrap.fill(answer))

