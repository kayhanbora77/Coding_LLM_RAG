import os
from dotenv import load_dotenv

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

# Load environment variables from .env file
load_dotenv()

import tiktoken
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from operator import itemgetter


# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # From console.groq.com
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")  # From huggingface.co
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVAL_K = 1
WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
CSS_CLASSES = ("post-content", "post-title", "post-header")


def setup_environment():
    """Set up environment variables required for the application."""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


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
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_vector_store(splits, embedding_model_name):
    """Create a vector store from document splits."""
    embeddings = HuggingFaceEndpointEmbeddings(model=embedding_model_name)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )


def create_retriever(vectorstore, k_value):
    """Create a retriever from the vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": k_value})


def create_generate_queries_chain():
    """Create a chain to generate multiple query perspectives."""
    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}"""

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    return (
        prompt_perspectives
        | ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs."""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def create_rag_chain(retriever, llm_model_name):
    """Create the complete RAG chain for question answering."""
    # Create the query generation chain
    generate_queries = create_generate_queries_chain()

    # Create the retrieval chain with multi-query approach
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    # Create the final RAG template
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(model=llm_model_name, temperature=0)

    return (
        {"context": retrieval_chain,
         "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )


def test_retriever(retriever):
    """Test the retriever with a sample question."""
    docs = retriever.invoke("What is Task Decomposition?")
    print(f"Number of documents retrieved: {len(docs)}\n")


def main():
    """Main function to execute the RAG pipeline."""
    print("Setting up environment...")
    setup_environment()

    print("Loading documents...")
    blog_docs = load_documents(WEB_PATHS, CSS_CLASSES)

    print("Splitting documents...")
    splits = split_documents(blog_docs, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating vector store...")
    vectorstore = create_vector_store(splits, EMBEDDING_MODEL_NAME)

    print("Creating retriever...")
    retriever = create_retriever(vectorstore, RETRIEVAL_K)

    print("Testing retriever...")
    test_retriever(retriever)

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(retriever, LLM_MODEL_NAME)

    print("Running the RAG pipeline...")
    question = "What is task decomposition for LLM agents?"

    print("What is task decomposition for LLM agents?")
    answer = rag_chain.invoke({"question": question})
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
