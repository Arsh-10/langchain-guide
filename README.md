# LangChain Guide

Welcome to the LangChain Guide! This repository serves as a comprehensive resource for anyone looking to dive into the world of LangChain, offering a step-by-step journey from understanding the basics to crafting impactful projects.

## Files Included:

### 1. `chat_completion.py`

This Python script demonstrates how to use LangChain for chat completion. By leveraging the OpenAI API, it allows users to input a section of text and receive a completed version based on the context and language patterns.

**Usage:**
```
python chat_completion.py
```

### 2. `review_sentiment_analysis.py`

The `review_sentiment_analysis.py` file contains code for sentiment analysis on dataframes from a CSV file. It utilizes LangChain to evaluate and classify the sentiment of each review as either negative or positive.

**Usage:**
```
python review_sentiment_analysis.py
```

### 3. `pdf_embeddings.py`

The `pdf_embeddings.py` script demonstrates how to use embeddings in LangChain with PDF files. It first loads the PDF using PyMuPDFLoader, uses CharacterTextSplitter to make chunk sizes, and employs the embedding model `intfloat` provided by HuggingFaceEmbeddings. It uses FAISS as a vectordb and finally performs a similarity search based on the query given by the user.

**Usage:**
```
python pdf_embeddings.py
```

### 4. `txt_embeddings.py`

The `txt_embeddings.py` script performs the same functionality as `pdf_embeddings.py` but is tailored for text (txt) files.

**Usage:**
```
python txt_embeddings.py
```

#### 5. `qa_pdf.py`

This Python script demonstrates how to process PDF files for Question Answering (QA) using LangChain. It loads a PDF document using PyMuPDFLoader, splits the text into chunks using CharacterTextSplitter, utilizes embeddings from HuggingFaceEmbeddings (`intfloat` model), creates vectors with FAISS, and executes a similarity search to retrieve answers based on the user query.

**Usage:**
```
python qa_pdf.py
```

#### 6. `qa_txt.py`

The `qa_txt.py` script performs similar functionality as `qa_pdf.py`, but it's tailored for text (txt) files. It loads a text file using TextLoader, splits the text into chunks using CharacterTextSplitter, applies embeddings from HuggingFaceEmbeddings (`intfloat` model), generates vectors using FAISS, and executes a similarity search to retrieve answers based on the user query.

**Usage:**
```
python qa_txt.py
```

#### 7. `qd_pdf.py`

The `qd_pdf.py` script illustrates the utilization of Qdrant, a cloud-based vector store platform, for processing PDF documents. It employs LangChain to load a PDF file through PyMuPDFLoader, segment the text into chunks using CharacterTextSplitter, employ embeddings from HuggingFaceEmbeddings (`sentence-transformers/all-mpnet-base-v2` model), create vectors in Qdrant using Qdrant's client, and execute a similarity search in the collection based on the provided query.

**Usage:**
```
python qd_pdf.py
```

These comments help users understand the functionalities and usage instructions for each script, enhancing the clarity and usability of the provided code files.

**Requirements:**
- Dependencies listed in `requirements.txt`

## Getting Started:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/langchain-guide.git
cd langchain-guide
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the scripts and explore LangChain functionalities:

```bash
python chat_completion.py
python review_sentiment_analysis.py
```

Feel free to explore, experiment, and contribute to the LangChain Guide! If you have any questions or suggestions, please open an issue or submit a pull request.

Happy coding!

---

Make sure to replace "your-username" with your actual GitHub username. Additionally, if there are more specific instructions or details for using these scripts, you can add them to the README file.
