# main.py
import streamlit as st
import pysqlite3
import sys, os
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import quote
from datetime import datetime
import logging
import time
from functools import lru_cache
from langchain.tools.base import BaseTool



# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize SQLite
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

# Data Models with validation
class Document(BaseModel):
    content: str = Field(..., description="Text content of the document")
    source: str = Field(..., description="Source URL or identifier")
    relevance_score: float = Field(0.0, description="Relevance score of the document")
    last_updated: datetime = Field(None, description="Last update timestamp")

    @classmethod
    def validate_content(cls, content: str) -> bool:
        """Validate document content."""
        return bool(content.strip()) and len(content.strip()) >= 10

class Documents(BaseModel):
    documents: List[Document]

    @classmethod
    def validate_documents(cls, documents: List[Document]) -> bool:
        """Validate document list."""
        return bool(documents) and all(doc.validate_content() for doc in documents)

class ComplianceAdvice(BaseModel):
    advice: str = Field(..., description="Generated compliance advice")
    sources: List[str] = Field(..., description="List of source references")

# Configuration with improved defaults
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CACHE_TTL = 3600  # 1 hour cache
    MAX_DOCUMENTS = 5
    TIMEOUT = 10  # seconds
    RETRY_COUNT = 3  # Number of retries for failed requests
    BACKOFF_FACTOR = 1.5  # Exponential backoff factor

# Initialize LLM with error handling
try:
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=Config.GROQ_API_KEY
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise RuntimeError("Failed to initialize LLM") from e

# Placeholder for relevance score calculation
def calculate_relevance_score(content: str, query: str) -> float:
    """Calculate relevance score between content and query."""
    # Implement your relevance scoring logic here
    return 0.5

# Document Retrieval Tool with retry mechanism
@st.cache_data(ttl=Config.CACHE_TTL)
def retrieve_documents(query: str) -> List[Document]:
    """Retrieve RBI compliance documents in real-time with retry mechanism."""
    try:
        encoded_query = quote(query)
        url = f"https://www.rbi.org.in/Scripts/SearchResults.aspx?search={encoded_query}"
        
        # Implement retry with exponential backoff
        for attempt in range(Config.RETRY_COUNT):
            try:
                response = requests.get(url, timeout=Config.TIMEOUT)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == Config.RETRY_COUNT - 1:
                    raise
                delay = Config.TIMEOUT * (Config.BACKOFF_FACTOR ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}/{Config.RETRY_COUNT}). "
                             f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        
        soup = BeautifulSoup(response.content, "html.parser")
        results = soup.find_all("div", class_="searchResultItem")
        
        if not results:
            logger.warning("No search results found")
            return []
        
        documents = []
        for result in results[:Config.MAX_DOCUMENTS]:
            title_elem = result.find("h3")
            if title_elem and title_elem.find("a"):
                title = title_elem.find("a").text.strip()
                link = title_elem.find("a")["href"]
            else:
                title = "No title"
                link = "#"
            
            snippet_elem = result.find("div", class_="searchResultSnippet")
            snippet = snippet_elem.text.strip() if snippet_elem else "No snippet"
            
            if Document.validate_content(snippet):
                documents.append(Document(
                    content=snippet,
                    source=link,
                    relevance_score=calculate_relevance_score(snippet, query)
                ))
        
        return documents
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve documents: {e}")
        raise

# Placeholder for document retrieval tool
class DocumentRetrievalTool(BaseTool):
    name: str = "retrieve_documents"
    description: str = "Retrieves documents from RBI website"

    def _run(self, query: str) -> List[Document]:
        return retrieve_documents(query)

    async def _arun(self, query: str) -> List[Document]:
        raise NotImplementedError()


# Agents and Tasks with improved error handling
def create_agents():
    """Create and configure CrewAI agents with error handling."""
    try:
        retrieval_agent = Agent(
            name="retrieval_agent",
            role="Document Retriever",
            goal="Retrieve relevant RBI documents in real-time",
            backstory="Specialized in fetching compliance data from RBI sources.",
            verbose=True,
            llm=llm
        )
        
        response_agent = Agent(
            role="Compliance Advisor",
            goal="Generate actionable RBI compliance advice",
            backstory="Expert in interpreting RBI regulations.",
            verbose=True,
            llm=llm
        )
        
        return retrieval_agent, response_agent
    except Exception as e:
        logger.error(f"Failed to create agents: {e}")
        raise

def create_tasks(retrieval_agent, response_agent):
    """Create and configure CrewAI tasks with validation."""
    try:
        retrieval_task = Task(
            description="Retrieve RBI documents relevant to the query",
            agent=retrieval_agent,
            expected_output="Documents",  # Fixed: should be string
            tools=[DocumentRetrievalTool()]  # Fixed: proper tool configuration
        )
        
        response_task = Task(
            description="Provide compliance advice based on retrieved documents",
            agent=response_agent,
            expected_output="ComplianceAdvice",  # Fixed: should be string
            context=[retrieval_task]
        )
        
        return retrieval_task, response_task
    except Exception as e:
        logger.error(f"Failed to create tasks: {e}")
        raise

# Placeholder for displaying results
def display_results(result):
    """Display the results in Streamlit."""
    st.write("Compliance Advice:")
    st.write(result["response_task"].output.advice)
    st.write("Sources:")
    for source in result["response_task"].output.sources:
        st.write(source)

# Main Application with improved error handling and user feedback
def main():
    """Main application entry point with enhanced error handling."""
    st.title("RBI Compliance Advisory Tool")
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .query-input {
            font-size: 16px;
            padding: 15px;
        }
        .result-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Query input with templates
    query = st.text_input(
        "Enter your compliance query:",
        placeholder="Type your query or select a template",
        help="Select from common templates or enter a custom query"
    )
    
    if st.button("Get Compliance Advice"):
        if not query.strip():
            st.warning("Please enter a valid query")
            return
        
        try:
            with st.spinner("Processing your query..."):
                # Initialize crew only when needed
                retrieval_agent, response_agent = create_agents()
                retrieval_task, response_task = create_tasks(retrieval_agent, response_agent)
                crew = Crew(
                    agents=[retrieval_agent, response_agent],
                    tasks=[retrieval_task, response_task],
                    verbose=True
                )
                
                result = crew.kickoff(inputs={"query": query})
                display_results(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
