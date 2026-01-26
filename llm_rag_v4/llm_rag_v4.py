import os

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads


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


def create_rag_chain(retriever, llm_model_name):
    """Create the complete RAG chain for question answering."""
    # RAG-Fusion: Related
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_rag_fusion
        | ChatGroq(model=llm_model_name, temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    # Create the retrieval chain with multi-query approach
    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

    # Create the final RAG template
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model=llm_model_name, temperature=0)

    return (
        {"context": retrieval_chain,
         "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )


def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


def main():
    """Main function to execute the RAG pipeline."""
    # Setup environment
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    print("Loading and processing documents...")
    splits = process_documents(WEB_PATHS, CSS_CLASSES, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating index...")
    vectorstore, retriever = create_index(splits, EMBEDDING_MODEL_NAME, RETRIEVAL_K)

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
