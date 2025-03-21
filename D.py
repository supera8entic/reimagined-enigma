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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize SQLite
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

# Data Models
class Document(BaseModel):
    content: str = Field(..., description="Text content of the document")
    source: str = Field(..., description="Source URL or identifier")
    relevance_score: float = Field(0.0, description="Relevance score of the document")
    last_updated: datetime = Field(None, description="Last update timestamp")

class Documents(BaseModel):
    documents: List[Document]

class ComplianceAdvice(BaseModel):
    advice: str = Field(..., description="Generated compliance advice")
    sources: List[str] = Field(..., description="List of source references")

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CACHE_TTL = 3600  # 1 hour cache
    MAX_DOCUMENTS = 5
    TIMEOUT = 10  # seconds

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=Config.GROQ_API_KEY
)

# Document Retrieval Tool
@st.cache_data(ttl=Config.CACHE_TTL)
def retrieve_documents(query: str) -> List[Document]:
    """Retrieve RBI compliance documents in real-time."""
    try:
        encoded_query = quote(query)
        url = f"https://www.rbi.org.in/Scripts/SearchResults.aspx?search={encoded_query}"
        
        response = requests.get(url, timeout=Config.TIMEOUT)
        response.raise_for_status()
        
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
            
            documents.append(Document(
                content=snippet,
                source=link,
                relevance_score=calculate_relevance_score(snippet, query)
            ))
        
        return documents
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve documents: {e}")
        return []

# Agents and Tasks
def create_agents():
    """Create and configure CrewAI agents."""
    retrieval_agent = Agent(
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

def create_tasks():
    """Create and configure CrewAI tasks."""
    retrieval_task = Task(
        description="Retrieve RBI documents relevant to the query",
        agent=retrieval_agent,
        expected_output=Documents,
        tools=[retrieve_documents_tool]
    )
    
    response_task = Task(
        description="Provide compliance advice based on retrieved documents",
        agent=response_agent,
        expected_output=ComplianceAdvice,
        context=[retrieval_task]
    )
    
    return retrieval_task, response_task

# Main Application
def main():
    """Main application entry point."""
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
                result = crew.kickoff(inputs={"query": query})
                display_results(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
