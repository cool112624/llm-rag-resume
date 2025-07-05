# rag_resume.py
# See project and latest version at: https://github.com/cool112624/llm-rag-resume

"""
LLM+RAG Resume Q&A Tool

- Loads a resume (PDF or TXT)
- Splits into semantic chunks
- Embeds and indexes with Sentence Transformers + FAISS
- Answers questions using OpenAI GPT-4o mini (RAG)
"""

# --- 1. Imports and Setup ---
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from getpass import getpass

# --- 2. Resume Loading ---
def load_resume(path: str) -> str:
    if path.lower().endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)
    elif path.lower().endswith('.txt'):
        with open(path, encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Please provide a PDF or TXT file.")

# --- 3. Chunking ---
def parse_resume_by_paragraph(text: str, min_length=40) -> list[str]:
    """Split text on double newlines, filter short paragraphs."""
    return [p.strip() for p in text.split('\n\n') if len(p.strip()) >= min_length]

# --- 4. Embedding + Indexing ---
def build_faiss_index(chunks: list[str]) -> tuple:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))
    return embedder, index

# --- 5. OpenAI API Key ---
def get_openai_client() -> openai.OpenAI:
    api_key = getpass("Enter your OpenAI API key: ")
    return openai.OpenAI(api_key=api_key)

# --- 6. RAG QA Function ---
def rag_qa(query: str, resume_chunks: list[str], embedder, index, client, top_k=3) -> str:
    query_emb = embedder.encode([query])
    _, I = index.search(np.array(query_emb), top_k)
    context = "\n\n".join([resume_chunks[i] for i in I[0]])
    prompt = f"Given the following context from my resume:\n\n{context}\n\nAnswer this question: {query}\n"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 7. CLI Main Loop ---
def main():
    print("="*60)
    print("LLM+RAG Resume Q&A Tool")
    print("Ask questions about your resume—AI answers are grounded in your uploaded document.")
    print("Project & updates: https://github.com/cool112624/llm-rag-resume")
    print("Model: OpenAI GPT-4o mini | Retrieval: Sentence Transformers + FAISS")
    print("="*60 + "\n")
    resume_path = input("Enter the path to your resume (PDF or TXT): ").strip()
    resume_text = load_resume(resume_path)
    resume_chunks = parse_resume_by_paragraph(resume_text)
    print(f"Parsed {len(resume_chunks)} chunks from resume.")
    embedder, index = build_faiss_index(resume_chunks)
    client = get_openai_client()
    print("\nAsk questions about your resume (type 'exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() == 'exit':
            break
        answer = rag_qa(query, resume_chunks, embedder, index, client)
        print("\n" + answer + "\n")

if __name__ == "__main__":
    main()
