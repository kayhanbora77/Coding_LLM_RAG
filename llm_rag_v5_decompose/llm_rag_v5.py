import os
from dotenv import load_dotenv

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

# Load environment variables from .env file
load_dotenv()

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
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


def format_qa_pair(question, answer):
    """Format Q and A pair"""
    formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain, retriever, llm):
    """RAG on each sub-question"""

    # 1. Decompose question
    sub_questions = sub_question_generator_chain.invoke({"question": question})

    # Initialize a list to hold RAG chain results
    rag_results = []

    # 2. Loop through sub-questions
    for sub_question in sub_questions:

        # Retrieve documents for each sub-question
        retrieved_docs = retriever.invoke(sub_question)

        # Format docs for the prompt (joining page content)
        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Use retrieved documents and sub-question in RAG chain
        answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": context_str,
                                                                 "question": sub_question})
        rag_results.append(answer)
        print(f"Answering sub-question: {sub_question}")

    return rag_results, sub_questions


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()


def main():
    """Main function to execute the RAG pipeline."""
    # Setup environment
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    print("Loading and processing documents...")
    splits = process_documents(WEB_PATHS, CSS_CLASSES, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating index...")
    vectorstore, retriever = create_index(splits, EMBEDDING_MODEL_NAME, RETRIEVAL_K)

    # ==========================================
    # 1. Decomposition Setup
    # ==========================================
    template_decomposition = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template_decomposition)

    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)

    # Chain to generate sub-questions
    generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # ==========================================
    # 2. RAG Prompt (Manual definition replacing hub.pull)
    # ==========================================
    # This replaces: prompt_rag = hub.pull("rlm/rag-prompt")
    prompt_rag = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Question: {question}
        Context: {context}
        Answer:"""
    )

    # ==========================================
    # 3. Execution
    # ==========================================
    question = "What are the main components of an LLM-powered autonomous agent system?"

    # Generate answers for each sub-question
    answers, questions = retrieve_and_rag(
        question,
        prompt_rag,
        generate_queries_decomposition,
        retriever,
        llm
    )

    # Format the pairs for the final synthesis
    context = format_qa_pairs(questions, answers)

    # ==========================================
    # 4. Synthesis
    # ==========================================
    template_final = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """
    prompt_final = ChatPromptTemplate.from_template(template_final)

    final_rag_chain = (
        prompt_final
        | llm
        | StrOutputParser()
    )

    print("\nGenerating Final Answer...\n")
    final_answer = final_rag_chain.invoke({"context": context, "question": question})
    print(f"Final Answer:\n{final_answer}")


if __name__ == "__main__":
    main()
