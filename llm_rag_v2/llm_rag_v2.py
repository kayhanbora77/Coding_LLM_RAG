import os

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

import tiktoken
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import numpy as np
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
GROQ_API_KEY = "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"  # From console.groq.com
HUGGINGFACEHUB_API_TOKEN = "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"  # From huggingface.co
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVAL_K = 1
WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
CSS_CLASSES = ("post-content", "post-title", "post-header")
ENCODING_NAME = "cl100k_base"


def setup_environment():
    """Set up environment variables required for the application."""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def format_docs(docs):
    """Joins the page content of the retrieved documents into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


def manual_embedding_test():
    """Test manual embedding similarity calculation."""
    question = "What kinds of pets do I like?"
    document = "My favorite pet is a cat."

    # Check tokens (optional)
    print(f"Tokens in question: {num_tokens_from_string(question, ENCODING_NAME)}")

    # Initialize Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Embed Query and Document
    query_result = embeddings.embed_query(question)
    document_result = embeddings.embed_query(document)

    # Calculate Similarity
    similarity = cosine_similarity(query_result, document_result)
    print(f"Cosine Similarity: {similarity}\n")


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


def create_rag_chain(retriever, prompt_template, llm_model_name):
    """Create the RAG chain for question answering."""
    llm = ChatGroq(model=llm_model_name, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
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

    print("Running manual embedding test...")
    manual_embedding_test()

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

    # Define Prompt
    template = """Answer the question based only on the following context:
{context}

Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    print("Creating RAG chain...")
    rag_chain = create_rag_chain(retriever, prompt, LLM_MODEL_NAME)

    print("Running the RAG pipeline...")
    print("Asking: 'What is Task Decomposition?'")
    answer = rag_chain.invoke("What is Task Decomposition?")
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
