# RAG Learning Bot v8: Logic Routing

An advanced Retrieval-Augmented Generation (RAG) system implementing logic-based routing for directing queries to appropriate data sources.

## Overview

This implementation uses a logic-based routing system where the system determines which data source would be most appropriate for answering a given question, then routes the query accordingly. This approach is particularly useful when dealing with multiple specialized knowledge bases.

## Features

- **Logic-Based Routing**: Determines optimal data source for each query
- **Structured Output**: Uses Pydantic models for consistent routing decisions
- **Language Detection**: Identifies programming language in queries
- **Specialized Chains**: Different processing chains for different data sources
- **Dynamic Query Routing**: Intelligent redirection based on content analysis

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Ensure you're in the project root directory
2. Install required packages (if not already installed):
   ```bash
   pip install -r ../requirements.txt
   pip install pydantic
   ```

## Configuration

Before running, configure your API keys in the `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

## Usage

Run the script directly:
```bash
python llm_rag_v8.py
```

The system will:
1. Load documents from the configured web source
2. Split documents into chunks and create vector store
3. Analyze the sample question to determine appropriate data source
4. Route the query based on programming language detection
5. Execute the appropriate processing chain

## How Logic Routing Works

The routing system involves several phases:

1. **Query Analysis**:
   - Input question: "Why doesn't the following code work: [Python code sample]"
   - Analyze for programming language indicators
   - Determine most relevant data source

2. **Structured Decision Making**:
   - Use LLM with structured output to make routing decision
   - Return consistent data structure with routing choice
   - Apply business logic based on decision

3. **Chain Execution**:
   - Execute the appropriate processing chain
   - Handle the query with specialized logic

## Architecture

The system implements:

- **Query Analyzer**: Detects language和技术 indicators in questions
- **Structured Router**: Makes routing decisions with Pydantic models
- **Chain Selector**: Chooses appropriate processing chain
- **Specialized Processors**: Different logic for different data sources

## Key Components

- **RouteQuery Model**: Pydantic model defining routing choices
- **Structured LLM**: Uses `with_structured_output` for consistent decisions
- **Routing Logic**: Business logic to determine next steps
- **Chain Executor**: Executes appropriate processing chain

## Customization

You can modify:

- **Routing Options**: Update `Literal` type in `RouteQuery` for different data sources
- **Detection Logic**: Modify `choose_route` function for different routing criteria
- **Data Sources**: Add new data source options to the routing model
- **Processing Chains**: Implement different logic for each data source
- **Model Configuration**: Adjust `LLM_MODEL_NAME` and `EMBEDDING_MODEL_NAME`

## Example Workflow

1. **Input**: Question with Python code sample
2. **Analysis**: System detects Python-related content
3. **Routing**: Routes to "python_docs" data source
4. **Execution**: Executes Python-specific processing chain
5. **Response**: Returns appropriate response based on routing

## Dependencies

Same as main project plus:
- `pydantic`: For structured output models
- `langchain-huggingface`
- `langchain-community`
- `langchain-core`
- `langchain-groq`
- `bs4`
- `chromadb`
- `langchain-text-splitters`

## Advanced Usage

Modify the routing logic to handle different types of queries:
```python
# In the RouteQuery class, change the Literal options:
datasource: Literal["technical_docs", "user_manuals", "api_docs"] = Field(...)
```

## Troubleshooting

- Ensure API keys are properly configured in `.env`
- Check that pydantic is installed: `pip install pydantic`
- Verify all dependencies are installed
- Confirm sufficient API quota for Groq requests

## Related Versions

This builds upon:
- **v7**: HyDE (Hypothetical Document Embeddings) for improved retrieval
- **v6**: Step-Back Prompting (enhanced reasoning)
- **v5**: Task Decomposition (handles complex multi-part questions)

## License

This project is open-source and available under the MIT License.
