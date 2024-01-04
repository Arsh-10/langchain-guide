#this is for pdf document 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

"""
    Loads a PDF file, extracts text chunks, embeds them using Hugging Face embeddings,
    and performs a similarity search with the given query using a FAISS vector database.

    Args:
    - file (str): The path to the PDF file.
    - query (str): The query for similarity search.

    Returns:
    None: Prints the document and its similarity score.

    Raises:
    - FileNotFoundError: If the specified file is not found.
    - Exception: For any other unexpected errors during the process.
    """
def pdf_loader(file, query):

    try:
        # Load the PDF file using PyMuPDFLoader
        loader = PyMuPDFLoader(file)
        documents = loader.load()

        # Split the text into chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=384, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Define the embedding model (intfloat/e5-small-v2 in this case)
        embed_model = "intfloat/e5-small-v2"
        embed = HuggingFaceEmbeddings(model_name=embed_model)

        # Create a FAISS vector database from the documents
        db = FAISS.from_documents(docs, embed)

        # Perform similarity search with the given query
        docs_and_scores = db.similarity_search_with_score(query)

        # Print the document and its similarity score
        print(docs_and_scores[0])

    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
pdf_loader('pdf_test.pdf', "In what format should student complete the product")