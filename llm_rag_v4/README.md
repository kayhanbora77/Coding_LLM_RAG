# RAG Learning Bot v4

An advanced Retrieval-Augmented Generation (RAG) system that implements RAG-Fusion and Reciprocal Rank Fusion techniques for improved question-answering capabilities using LangChain, Groq, and Hugging Face.

## Overview

This RAG system retrieves information from web documents and uses a Large Language Model to answer questions about the content. The system implements advanced retrieval techniques including multi-query generation and Reciprocal Rank Fusion to improve answer quality and relevance.

## Features

- **Web Document Loading**: Fetches content from specified URLs using WebBaseLoader
- **Intelligent Chunking**: Splits documents into manageable chunks for processing
- **Vector Storage**: Uses Chroma for efficient similarity search
- **RAG-Fusion**: Generates multiple query variations to improve retrieval
- **Reciprocal Rank Fusion (RRF)**: Ranks and fuses results from multiple queries for better relevance
- **Deduplication**: Removes duplicate retrieved documents
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
   cd llm_rag_v4
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the required packages individually:
   ```bash
   pip install langchain-huggingface langchain-community langchain-core langchain-groq langchain-text-splitters bs4 chromadb
   ```

## Configuration

Before running the application, you need to configure your API keys in the `llm_rag_v4.py` file:

1. Replace `GROQ_API_KEY` with your actual Groq API key
2. Replace `HUGGINGFACEHUB_API_TOKEN` with your actual Hugging Face token

## Usage

Run the script directly:
```bash
python3 llm_rag_v4.py
```

The system will:
1. Load documents from the configured web sources
2. Split documents into chunks
3. Create a vector store for similarity search
4. Generate multiple query variations using RAG-Fusion
5. Apply Reciprocal Rank Fusion to rank results
6. Generate answers to your questions using the enhanced RAG pipeline

## Architecture

The system is composed of several key components:

- **Document Loader**: Fetches content from web sources using BeautifulSoup filtering
- **Text Splitter**: Breaks documents into smaller chunks for vector storage
- **Embedding Model**: Converts text to vectors using Hugging Face embeddings
- **Vector Store**: Chroma database for efficient similarity search
- **Query Generator**: Creates multiple query variations to improve retrieval
- **RRF Processor**: Applies Reciprocal Rank Fusion to enhance result ranking
- **Retriever**: Fetches relevant documents based on the query
- **LLM Chain**: Combines retrieved context with user questions for answer generation

## Advanced Techniques

### RAG-Fusion
The system generates multiple query variations to capture different perspectives of the user's question, improving retrieval coverage.

### Reciprocal Rank Fusion (RRF)
RRF is used to rerank and fuse results from multiple queries, providing more relevant results by considering rankings from all query variations.

## Customization

You can customize the following aspects:

- **Data Sources**: Modify `WEB_PATHS` to load content from different URLs
- **Content Filtering**: Update `CSS_CLASSES` to target specific HTML elements
- **Chunking Strategy**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters
- **Models**: Change `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` to use different models
- **Retrieval Settings**: Modify `RETRIEVAL_K` to control the number of documents retrieved
- **RRF Parameter**: Adjust the `k` value in `reciprocal_rank_fusion` for different ranking behaviors

## Dependencies

- `langchain-huggingface`: For Hugging Face embeddings
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database
- `langchain-text-splitters`: Text chunking utilities
- `langchain_core.load`: Utilities for document serialization

## How It Works

1. **Indexing Phase**:
   - Load web documents
   - Split into chunks
   - Create embeddings
   - Store in vector database

2. **Query Phase**:
   - Generate multiple query perspectives using RAG-Fusion
   - Retrieve documents for each query
   - Apply Reciprocal Rank Fusion to rerank results
   - Combine context with question
   - Generate answer using LLM

## Example Output

The system will output the final answer to the question "What is task decomposition for LLM agents?"

## Troubleshooting

- Ensure your API keys are valid and have sufficient quota
- Check that all dependencies are installed correctly
- Verify that the target websites are accessible
- Make sure the CSS selectors match the target website structure

## License

This project is open-source and available under the MIT License.
