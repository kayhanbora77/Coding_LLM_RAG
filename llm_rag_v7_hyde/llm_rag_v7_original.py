from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # From console.groq.com
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")  # From huggingface.co
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4  # Increased to 4 to get better context for sub-questions
WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
CSS_CLASSES = ("post-content", "post-title", "post-header")


def process_documents(web_paths, css_classes, chunk_size, chunk_overlap):
    """Load and process documents from web sources."""
    # Load documents
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=css_classes
            )
        ),
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    return splits


def create_index(splits, embedding_model_name, k_value):
    """Create vector store and retriever from document splits."""
    # Create vector store
    embeddings = HuggingFaceEndpointEmbeddings(model=embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    return vectorstore, retriever


def main():
    """Main function to execute the RAG pipeline."""
    # Setup environment
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    print("Loading and processing documents...")
    splits = process_documents(WEB_PATHS, CSS_CLASSES, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating index...")
    vectorstore, retriever = create_index(splits, EMBEDDING_MODEL_NAME, RETRIEVAL_K)

    # HyDE document generation
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)


    generate_docs_for_retrieval = (
        prompt_hyde | ChatGroq(model=LLM_MODEL_NAME, temperature=0) | StrOutputParser()
    )

    # Run
    question = "What is task decomposition for LLM agents?"
    generate_docs_for_retrieval.invoke({"question":question})

    # Retrieve
    retrieval_chain = generate_docs_for_retrieval | retriever
    retrieved_docs = retrieval_chain.invoke({"question":question})

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        prompt
        | ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        | StrOutputParser()
    )


    print("\nGenerating Answer...\n")
    final_answer = final_rag_chain.invoke({"context":retrieved_docs,"question":question})
    print(f"Answer:\n{final_answer}")

if __name__ == "__main__":
    main()
