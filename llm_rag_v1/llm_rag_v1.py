import os
from dotenv import load_dotenv

from bs4.filter import SoupStrainer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
CSS_CLASSES = ("post-content", "post-title", "post-header")


def setup_environment():
    """Set up environment variables required for the application."""
    os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


def format_docs(docs):
    """Joins the page content of the retrieved documents into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents(web_paths, css_classes):
    """Load documents from web sources using WebBaseLoader."""
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=css_classes
            )
        ),
    )
    return loader.load()


def split_documents(docs, chunk_size, chunk_overlap):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_vector_store(splits, embedding_model_name):
    """Create a vector store from document splits."""
    embeddings = HuggingFaceEndpointEmbeddings(model=embedding_model_name)
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def create_retriever(vectorstore):
    """Create a retriever from the vector store."""
    return vectorstore.as_retriever()


def create_rag_chain(retriever, llm_model_name):
    """Create the RAG chain for question answering."""
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model=llm_model_name, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    """Main function to execute the RAG pipeline."""
    print("Setting up environment...")
    setup_environment()

    print("Loading documents...")
    docs = load_documents(WEB_PATHS, CSS_CLASSES)

    print("Splitting documents...")
    splits = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating vector store...")
    vectorstore = create_vector_store(splits, EMBEDDING_MODEL_NAME)

    print("Creating retriever...")
    retriever = create_retriever(vectorstore)

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(retriever, LLM_MODEL_NAME)

    print("Loading data and building memory...")
    print("\nAsking the AI: 'What is Task Decomposition?'\n")

    answer = rag_chain.invoke("What is Task Decomposition?")
    print(answer)


if __name__ == "__main__":
    main()
