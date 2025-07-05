
# LLM-RAG-Resume

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Issues](https://img.shields.io/github/issues/cool112624/llm-rag-resume)
![Last Commit](https://img.shields.io/github/last-commit/cool112624/llm-rag-resume)

---

## 🚀 Overview

**LLM-RAG-Resume** is a Retrieval-Augmented Generation (RAG) powered Q&A tool for any resume.  
Ask questions about a resume (yours or a sample), and get AI-generated answers—grounded in the actual resume content—using OpenAI’s GPT-4o mini.

---

## ✨ Features

- Fast, semantic search over any resume (PDF or TXT)
- Accurate answers using OpenAI GPT-4o mini (or your preferred LLM)
- Uses FAISS for vector search and Sentence Transformers for embeddings
- Includes sample resumes for safe testing
- Command-line interface for interactive Q&A
- No internet retrieval—answers only use your provided resume content

---

## ⚡️ Quickstart

```bash
Option A - Local (Terminal / CLI)
# 1. Clone the repository
git clone https://github.com/yourname/llm-rag-resume.git
cd llm-rag-resume

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main script
python rag_resume.py
# - Enter the path to your resume file (e.g., sample_resume.txt)
# - Enter your OpenAI API key when prompted
# - Ask questions in your terminal

Option B - Notebook (Jupyter / Google Colab)
# 1. Open rag_resume.ipynb in Jupyter or Google Colab
# 2. Follow the instructions in the notebook cells
#    (You’ll be prompted to install dependencies, upload a resume, enter your API key, and start chatting)
```

---

## 📝 Usage

- Upload your own resume (PDF or TXT) to the repo folder, or use the included sample resumes.
- When prompted, enter your [OpenAI API key](https://platform.openai.com/api-keys).
- Ask questions interactively in your terminal, e.g.:
  - "What are the candidate’s technical strengths?"
  - "What projects has the candidate completed using Python?"

---

## 🗂 Project Structure

```text
llm-rag-resume/
├── rag_resume.py
├── rag_resume.ipynb
├── requirements.txt
├── README.md
├── sample_resume.pdf
├── sample_resume.txt
└── .gitignore
```

---

## 🧑‍💻 How it Works

1. **Resume Parsing:** Splits resume into semantic chunks (by paragraph).
2. **Embedding:** Generates embeddings using Sentence Transformers.
3. **Retrieval:** Uses FAISS to find the most relevant chunks for a question.
4. **LLM Answering:** Sends context+question to GPT-4o mini, which answers using only the retrieved text.

---

## 📚 Citation & Acknowledgement

- [Sentence Transformers](https://www.sbert.net/) ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) ([Johnson et al., 2017](https://arxiv.org/abs/1702.08734)) for vector search
- [OpenAI GPT-4o mini](https://platform.openai.com/docs/models/gpt-4o) for LLM generation

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋‍♂️ Contact

- For issues or feature requests, please open a [GitHub issue](https://github.com/yourname/llm-rag-resume/issues).
- For questions, opportunities, or feedback, please fill out [this contact form](https://docs.google.com/forms/d/e/1FAIpQLSfRIMDgGXS8VLHJlp8IPwDT34I0F-RrjLZXe3BWQhyO8jApVg/viewform?usp=dialog).

---
