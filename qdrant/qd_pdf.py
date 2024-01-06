from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

def pdf_loader(file, query):
    """
    This function demonstrates how to use Qdrant, a cloud-based platform for storing vector stores. 
    It loads a PDF document, splits its content into chunks, embeds the chunks, creates a Qdrant collection,
    stores vector stores in that collection, and retrieves data from the collection based on the provided query.
    
    Args:
    - file (str): Path to the PDF file.
    - query (str): Query used for similarity search in the Qdrant collection.
    """
    try:
        # Loading PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(file)
        documents = loader.load()
        
        # Splitting content of PDF into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        # Initializing HuggingFace embeddings model
        embed_model = "sentence-transformers/all-mpnet-base-v2"
        embed = HuggingFaceEmbeddings(model_name=embed_model)
        
        # Initializing Qdrant client
        qdrant_url = "#"
        qdrant_key = "#"  # Add your Qdrant API key here
        qdrant = Qdrant.from_documents(
            docs,
            embed,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_key,
            collection_name="my_documents",
        )
        
        # Performing similarity search in Qdrant collection based on the query
        found_docs = qdrant.similarity_search_with_score(query)
        document, score = found_docs[0]
        
        # Printing retrieved document content and score
        print(document.page_content)
        print(f"\nScore: {score}")

    except Exception as e:
        # Handling exceptions if any error occurs during the process
        print(f"An error occurred: {e}")

# Example usage: Provide the PDF file and query for similarity search
pdf_loader("tl.pdf", "What is transfer learning")
