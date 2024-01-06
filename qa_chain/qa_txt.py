import os
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Loading environment variables from the .env file
load_dotenv()

# Initializing OpenAI key
openai.api_key = os.getenv('openAI')

def qa_txt(file, query):
    """
    Process a text file (.txt) to extract text, split it into chunks, generate embeddings, and perform QA retrieval.
    
    Args:
    - file (str): Path to the text file.
    - query (str): Query for the QA system.
    """
    try:
        # Loading the text file using TextLoader
        loader = TextLoader(file)
        documents = loader.load()
        
        # Splitting the text into chunks using CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator='.')
        texts = text_splitter.split_documents(documents)
        
        # Initializing the HuggingFace embeddings model
        embed_model = "intfloat/e5-small-v2"
        embed = HuggingFaceEmbeddings(model_name=embed_model) 
        
        # Creating vectors using FAISS
        docsearch = FAISS.from_documents(texts, embed)

        # Defining a prompt template for the QA process
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. The answer shouldn't be more than 20 words.
        
        {context}
        
        Question: {question}
        Answer in English:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=['context', 'question']
        )

        # Setting up the QA chain for retrieval
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
        
        # Running the QA chain to retrieve an answer for the query
        ans = qa.run(query)
        print(ans)

    except Exception as e:
        # Handling exceptions if any error occurs during the process
        print(f"An error occurred: {e}")

# Example usage: Provide the text file and query for QA
qa_txt("tl.txt", "What is transfer learning")
