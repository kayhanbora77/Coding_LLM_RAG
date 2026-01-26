# RAG Learning Bot v2

An intermediate Retrieval-Augmented Generation (RAG) system that demonstrates embedding techniques and similarity calculations along with a complete RAG pipeline using LangChain, Groq, and Hugging Face.

## Overview

This RAG system combines manual embedding similarity testing with a complete RAG pipeline. It showcases how to calculate cosine similarity between embeddings and implements a full question-answering system based on external documents.

## Features

- **Manual Embedding Test**: Demonstrates cosine similarity calculation between query and document embeddings
- **Web Document Loading**: Fetches content from specified URLs using WebBaseLoader
- **Intelligent Chunking**: Splits documents into manageable chunks for processing
- **Vector Storage**: Uses Chroma for efficient similarity search
- **Similarity Calculation**: Implements cosine similarity for manual testing
- **Complete RAG Pipeline**: Full question-answering system with context retrieval
- **Modular Architecture**: Clean separation of concerns with dedicated functions

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm_rag_v2
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the required packages individually:
   ```bash
   pip install langchain-huggingface langchain-community langchain-core langchain-groq langchain-text-splitters tiktoken bs4 chromadb numpy
   ```

## Configuration

Before running the application, you need to configure your API keys in the `llm_rag_v2.py` file:

1. Replace `GROQ_API_KEY` with your actual Groq API key
2. Replace `HUGGINGFACEHUB_API_TOKEN` with your actual Hugging Face token

## Usage

Run the script directly:
```bash
python3 llm_rag_v2.py
```

The system will:
1. Perform a manual embedding similarity test
2. Load documents from the configured web sources
3. Split documents into chunks
4. Create a vector store for similarity search
5. Test the retriever with a sample question
6. Generate answers to your questions using the RAG pipeline

## Architecture

The system is composed of several key components:

- **Document Loader**: Fetches content from web sources using BeautifulSoup filtering
- **Text Splitter**: Breaks documents into smaller chunks for vector storage
- **Embedding Model**: Converts text to vectors using Hugging Face embeddings
- **Vector Store**: Chroma database for efficient similarity search
- **Similarity Calculator**: Computes cosine similarity between vectors
- **Retriever**: Fetches relevant documents based on the query
- **LLM Chain**: Combines retrieved context with user questions for answer generation

## Customization

You can customize the following aspects:

- **Data Sources**: Modify `WEB_PATHS` to load content from different URLs
- **Content Filtering**: Update `CSS_CLASSES` to target specific HTML elements
- **Chunking Strategy**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters
- **Models**: Change `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` to use different models
- **Retrieval Settings**: Modify `RETRIEVAL_K` to control the number of documents retrieved

## Dependencies

- `langchain-huggingface`: For Hugging Face embeddings
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `tiktoken`: Tokenization utilities
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database
- `numpy`: Numerical computing for similarity calculations
- `langchain-text-splitters`: Text chunking utilities

## How It Works

1. **Manual Embedding Test**:
   - Embed a sample question and document
   - Calculate cosine similarity between them

2. **Indexing Phase**:
   - Load web documents
   - Split into chunks
   - Create embeddings
   - Store in vector database

3. **Query Phase**:
   - Retrieve relevant documents
   - Combine context with question
   - Generate answer using LLM

## Example Output

The system will output the cosine similarity between test embeddings and information about the number of documents retrieved, followed by the final answer to the question "What is Task Decomposition?"

## Troubleshooting

- Ensure your API keys are valid and have sufficient quota
- Check that all dependencies are installed correctly
- Verify that the target websites are accessible
- Make sure the CSS selectors match the target website structure

## License

This project is open-source and available under the MIT License.
