# Coding LLM RAG Project

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations demonstrating progressive complexity and advanced techniques using LangChain, Groq, and Hugging Face.

## Overview

This project contains multiple versions of RAG systems, each building upon the previous with increasingly sophisticated techniques:

- **llm_rag_v1**: Basic RAG implementation with document loading and simple question answering
- **llm_rag_v2**: Enhanced with manual embedding similarity testing
- **llm_rag_v3**: Advanced multi-query approach for improved retrieval
- **llm_rag_v4**: Implementation of RAG-Fusion and Reciprocal Rank Fusion
- **llm_rag_v5**: Task Decomposition for handling complex questions
- **llm_rag_v6**: Step-Back Prompting for enhanced reasoning

## Features

- **Progressive Learning**: Each version introduces new concepts and techniques
- **Multiple Approaches**: Various RAG methodologies and enhancements
- **Modular Architecture**: Clean, maintainable code structure
- **Comprehensive Documentation**: Each version includes detailed README
- **Best Practices**: Following Python and LangChain best practices
- **Secure Configuration**: Environment-based API key management

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Coding_LLM_RAG
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure API keys:
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your actual API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

## Project Structure

```
Coding_LLM_RAG/
├── .env                   # API keys (not committed to git)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── llm_rag_v1/          # Basic RAG implementation
│   ├── llm_rag_v1.py
│   └── README.md
├── llm_rag_v2/          # With manual embedding tests
│   ├── llm_rag_v2.py
│   └── README.md
├── llm_rag_v3/          # Multi-query enhancement
│   ├── llm_rag_v3.py
│   └── README.md
├── llm_rag_v4/          # RAG-Fusion and Reciprocal Rank Fusion
│   ├── llm_rag_v4.py
│   └── README.md
├── llm_rag_v5_decompose/ # Task Decomposition
│   ├── llm_rag_v5.py
│   ├── requirements.txt
│   └── README.md
└── llm_rag_v6_stepback/  # Step-Back Prompting
    ├── llm_rag_v6.py
    └── README.md
```

## Usage

Each version can be run independently:

```bash
# Navigate to the desired version directory
cd llm_rag_v1
python llm_rag_v1.py

# Or for other versions
cd ../llm_rag_v2
python llm_rag_v2.py

# For v6 with step-back prompting
cd ../llm_rag_v6_stepback
python llm_rag_v6.py
```

## Version Descriptions

### llm_rag_v1: Basic RAG
- Simple document loading and retrieval
- Basic question answering with context
- Foundational RAG concepts
- Uses web scraping to load content

### llm_rag_v2: With Embedding Tests
- Manual embedding similarity testing
- Cosine similarity calculations
- Enhanced with basic similarity metrics
- Demonstrates embedding concepts

### llm_rag_v3: Multi-Query Enhancement
- Generates multiple query perspectives
- Improved retrieval through query diversification
- Better handling of ambiguous questions
- Uses LLM to generate alternative queries

### llm_rag_v4: RAG-Fusion and RRF
- RAG-Fusion with multiple query generation
- Reciprocal Rank Fusion for result ranking
- Advanced retrieval enhancement techniques
- Combines multiple retrieval strategies

### llm_rag_v5: Task Decomposition
- Complex question breakdown into sub-questions
- Individual processing of sub-questions
- Synthesis of answers from multiple sub-answers
- Handles multi-part questions effectively

### llm_rag_v6: Step-Back Prompting
- Step-back prompting for enhanced reasoning
- Generates more generic questions first
- Uses context from both original and step-back queries
- Improved handling of complex reasoning tasks

## Dependencies

The project uses various LangChain components and external services:

- `langchain-huggingface`: For Hugging Face embeddings and models
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database for document storage
- `langchain-text-splitters`: Text chunking utilities
- `tiktoken`: Tokenization utilities
- `numpy`: Numerical computing for similarity calculations

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Groq API Key - Get from https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Hugging Face Token - Get from https://huggingface.co/settings/tokens
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

### Customization Options

Each version can be customized:

- **Data Sources**: Modify URLs and CSS selectors for different content
- **Models**: Change embedding and LLM models in configuration constants
- **Chunking Strategy**: Adjust chunk sizes and overlap parameters
- **Retrieval Settings**: Modify number of retrieved documents (k value)
- **Prompts**: Customize prompt templates for different behaviors

Example configuration in each script:
```python
# Configuration constants
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4
```

## Security Best Practices

- API keys are stored in `.env` file and excluded from git
- `.gitignore` prevents accidental commits of sensitive data
- Environment variables are used instead of hardcoded keys
- Regular key rotation is recommended

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Verify your API keys in `.env` file
   - Check that keys have proper permissions
   - Ensure keys haven't expired

2. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Model Access Issues**
   - Verify Hugging Face token permissions
   - Some models may require special access on Hugging Face

4. **Rate Limiting**
   - Free API tiers may have rate limits
   - Consider upgrading to paid plans for production use

### Debugging Tips

```bash
# Check Python environment
python --version
pip list

# Test API connectivity
python -c "import os; print('GROQ_API_KEY exists:', bool(os.getenv('GROQ_API_KEY')))"
python -c "import os; print('HUGGINGFACEHUB_API_TOKEN exists:', bool(os.getenv('HUGGINGFACEHUB_API_TOKEN')))"
```

## Contributing

Feel free to contribute by:
- Adding new RAG techniques and approaches
- Improving existing implementations
- Adding more comprehensive examples
- Enhancing documentation
- Reporting bugs and issues

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- LangChain community for excellent RAG tools
- Hugging Face for embedding models and infrastructure
- Groq for fast LLM inference
- ChromaDB for vector database capabilities

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Models](https://huggingface.co/models)
- [Groq Documentation](https://console.groq.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)