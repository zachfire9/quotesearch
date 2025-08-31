# Quote Search Application

A Python application that uses LangChain and local embeddings to search through a collection of quotes, with Claude for enhanced search capabilities.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Usage

Run the application:
```bash
python quote_search.py
```

Enter your search query when prompted. The application will return the most relevant quotes based on semantic similarity.

## How It Works

The application provides semantic search capabilities for your quotes collection:

1. **Data Loading**: Loads quotes from `quotes.json`
2. **Embedding Generation**:
   - Uses the `all-MiniLM-L6-v2` model from Hugging Face
   - Generates vector embeddings for each quote
   - The model will be downloaded automatically on first run (~80MB)
3. **Vector Search**:
   - Utilizes FAISS (Facebook AI Similarity Search) for fast similarity search
   - Indexes all quote embeddings for efficient retrieval
4. **Search Interface**:
   - Accepts natural language queries
   - Returns quotes ranked by semantic similarity
   - Displays a relevance percentage for each result

## First Run

On first launch, the application will:
1. Download the `all-MiniLM-L6-v2` embedding model
2. Process and index all quotes
3. This may take a few minutes depending on your internet connection

Subsequent runs will be much faster as the model is cached locally.

Type 'exit' to quit the application.
