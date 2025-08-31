import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

class QuoteSearcher:
    def __init__(self, json_path):
        """Initialize the QuoteSearcher with the path to quotes.json"""
        self.json_path = Path(json_path)
        self.vectorstore = None
        self._load_quotes()
    
    def _load_quotes(self):
        """Load quotes from JSON and create vector store"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract quote texts
        quotes = [quote["text"] for quote in data["quotes"]]
        
        # Create documents
        documents = [Document(page_content=quote) for quote in quotes]
        
        # Split documents into chunks (though quotes are already individual)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(documents)
        
        # Initialize embeddings (using a local model for embeddings)
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embeddings
        )
    
    def search_quotes(self, query: str, k: int = 5) -> list:
        """
        Search for quotes similar to the query
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (quote, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        # Get similar documents
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in docs:
            results.append({
                "quote": doc.page_content,
                "score": float(score)
            })
            
        return results

def main():
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please create a .env file with your Anthropic API key")
        return
    
    # Initialize searcher
    searcher = QuoteSearcher("quotes.json")
    print("Quote search initialized successfully!")
    print("Type 'exit' to quit.\n")
    
    # Interactive search loop
    print("\nEnter your search query (or type 'exit', 'quit', 'q' to quit):")
    while True:
        try:
            query = input("\nSearch: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting quote search. Goodbye!")
                break
                
            if not query:
                print("Please enter a search query or type 'exit' to quit.")
                continue
            
            results = searcher.search_quotes(query)
            
            if not results:
                print("No matching quotes found.\n")
                continue
                
            print("\nTop matching quotes (type 'exit' to quit):")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['quote']} (Relevance: {1 - result['score']/2:.1%}")
            print("\n" + "-"*80)  # Add a separator line
            
        except KeyboardInterrupt:
            print("\n\nExiting quote search. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
