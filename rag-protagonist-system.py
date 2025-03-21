import os
import re
from typing import List, Dict, Any


import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# Load environment variables (store your OPENAI_API_KEY in a .env file)
load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
#import openai
openai_api_key = "INSERT YOUR KEY HERE"
os.environ["OPENAI_API_KEY"] = openai_api_key  # Set environment variable
openai.api_key = openai_api_key


class FanFictionRAG:
    def __init__(self, character_name: str, pdf_directory: str):
        """
        Initialize the RAG system for fan fiction character roleplay.
        
        Args:
            character_name: Name of the main protagonist to roleplay as
            pdf_directory: Directory containing PDF files of the fan fiction
        """
        self.character_name = character_name
        self.pdf_directory = pdf_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.character_info = {}
        
        # Initialize system by loading and processing PDFs
        self.load_documents()
    
    def load_documents(self):
        """Load PDF documents, process them, and create a vector store."""
        documents = []
        
        # Load all PDFs from the directory
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        if not documents:
            raise ValueError(f"No PDF documents found in {self.pdf_directory}")
        
        # Extract character information while processing documents
        self.extract_character_info(documents)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"Loaded {len(chunks)} text chunks into vector store")
    
    def extract_character_info(self, documents: List[Document]):
        """
        Extract basic information about the main character.
        This is a simple implementation - you might want to use more sophisticated NLP for this.
        
        Args:
            documents: List of Document objects containing the fan fiction text
        """
        full_text = " ".join([doc.page_content for doc in documents])
        
        # Look for common character description patterns
        name_match = re.search(f"{self.character_name}(?:'s| is| was)(.+?)(?:\.|,)", full_text)
        if name_match:
            self.character_info["description"] = name_match.group(1).strip()
        
        # This is very basic - you would want more sophisticated extraction in a real system
        self.character_info["appears_in"] = [os.path.basename(doc.metadata.get("source", "")) 
                                           for doc in documents if self.character_name in doc.page_content]
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: The user's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call load_documents first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def create_prompt(self, query: str, relevant_docs: List[Document]) -> str:
        """
        Create a prompt for the LLM based on the query and retrieved documents.
        
        Args:
            query: The user's question
            relevant_docs: Retrieved document chunks
            
        Returns:
            Formatted prompt string
        """
        # Combine the relevant document chunks
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create a basic character description from extracted info
        character_desc = f"Name: {self.character_name}"
        if "description" in self.character_info:
            character_desc += f"\nDescription: {self.character_info['description']}"
        
        # Create the prompt template
        prompt_template = PromptTemplate(
            input_variables=["character_desc", "context", "query"],
            template="""
You are roleplaying as a character from a fan fiction story with the following details:

{character_desc}

Use ONLY the following context from the fan fiction to respond. If the information 
isn't in the context, say you don't recall or don't know about that, rather than making up information:

{context}

Respond in first person, as if you are the character. Stay true to the character's personality, 
knowledge, speech patterns, and background as depicted in the fan fiction.

User's question: {query}

Your response as {character_name}:
"""
        )
        
        # Fill in the template
        return prompt_template.format(
            character_desc=character_desc,
            context=context,
            query=query,
            character_name=self.character_name
        )
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the user's query by retrieving relevant context
        and using the OpenAI API.
        
        Args:
            query: The user's question
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Retrieve relevant documents
        relevant_docs = self.search_documents(query)
        
        # Create the prompt
        prompt = self.create_prompt(query, relevant_docs)
        
        # Generate response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use an appropriate model
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Return the response along with metadata
        return {
            "response": response.choices[0].message.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs],
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag_system = FanFictionRAG(
        character_name="El",  # Replace with your protagonist's name
        pdf_directory= r"D:\RAG\Tinkering with Life"  # Replace with your PDF directory
    )
    
    # Example interaction loop
    print(f"Ask {rag_system.character_name} a question (or type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        try:
            result = rag_system.generate_response(user_input)
            print("\n" + result["response"] + "\n")
            print(f"Sources: {', '.join([os.path.basename(src) for src in result['sources']])}")
        except Exception as e:
            print(f"Error: {e}")