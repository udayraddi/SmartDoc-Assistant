# SmartDoc-Assistant
An intelligent chatbot that helps you search, analyze, and retrieve key insights from documents.

## Overview

Document Buddy App is an AI-powered chatbot that allows users to upload PDF documents, generate embeddings, and interact with the content using a Retrieval-Augmented Generation (RAG) approach. It leverages Streamlit for the frontend, Qdrant for vector storage, and Llama 3.2 for LLM-based query handling.

## Features

- Upload PDF documents for processing.
- Generate embeddings using `BAAI/bge-small-en`.
- Store and retrieve embeddings from Qdrant.
- Interact with the chatbot for document-based queries using Llama 3.2.
- Implements query rewriting and retrieval enhancement using hypothetical document generation.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file and add the required keys:

```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

Run the application using:

```bash
streamlit run app.py
```

## Modules Description

### `app.py`

- Initializes Streamlit UI.
- Handles PDF upload and embedding generation.
- Manages chatbot interactions.

### `chatbot.py`

- Implements `ChatbotManager` class.
- Handles query rewriting, document retrieval, and final answer generation.
- Interacts with Groq LLM and Qdrant for RAG-based responses.

### `vectors.py`

- Implements `EmbeddingsManager` for handling document embeddings.
- Uses `BAAI/bge-small-en` for embedding generation.
- Cleans and preprocesses text before storing in Qdrant.

## How It Works

1. Upload a PDF in the Streamlit UI.
2. Click "Generate Embeddings" to process the document.
3. Start chatting with the chatbot about the document.

## Technologies Used

- **Streamlit**: Web interface
- **Hugging Face Embeddings**: Text embeddings
- **Qdrant**: Vector database
- **Llama 3.2 (via Groq)**: LLM for responses
- **LangChain**: Query and retrieval pipeline

## Future Improvements

- Support for multiple document formats (PPTX, DOCX, etc.).
- Improved query understanding with advanced retrieval techniques.
- Integration with more LLMs and embeddings models.

## License

MIT License

.




