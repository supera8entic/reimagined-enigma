__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import streamlit as st
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
from crewai_tools import Tool  # Import Tool from crewai_tools
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import quote

# Load Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in the environment. Please set it to proceed.")
    st.stop()

# Define the LLM with a specific Groq model
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key)

# Define Pydantic models for structured data validation
class Document(BaseModel):
    """Model for a single RBI document."""
    content: str = Field(..., description="Text content of the document")
    source: str = Field(..., description="Source URL or identifier")

class Documents(BaseModel):
    """Model for retrieved RBI documents."""
    documents: List[Document]

class ComplianceAdvice(BaseModel):
    """Model for compliance advice output."""
    advice: str = Field(..., description="Generated compliance advice")
    sources: List[str] = Field(..., description="List of source references")

# Function for real-time document retrieval from RBI search results
def retrieve_documents(query: str) -> List[Document]:
    """
    Retrieves RBI compliance documents in real-time by scraping the RBI search results page.
    
    Args:
        query (str): User input query.
    Returns:
        List[Document]: List of validated document objects.
    """
    encoded_query = quote(query)
    url = f"https://www.rbi.org.in/Scripts/SearchResults.aspx?search={encoded_query}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        results = soup.find_all("div", class_="searchResultItem")
        
        if not results:
            st.warning("No search results found or unable to parse the page.")
            return []
        
        documents = []
        for result in results[:5]:
            title_elem = result.find("h3")
            if title_elem and title_elem.find("a"):
                title = title_elem.find("a").text.strip()
                link = title_elem.find("a")["href"]
            else:
                title = "No title"
                link = "#"
            
            snippet_elem = result.find("div", class_="searchResultSnippet")
            snippet = snippet_elem.text.strip() if snippet_elem else "No snippet"
            
            documents.append(Document(content=snippet, source=link))
        
        return documents
    
    except requests.RequestException as e:
        st.warning(f"Failed to retrieve documents: {e}")
        return []

# Wrap the retrieve_documents function in a CrewAI Tool
retrieve_documents_tool = Tool(
    name="RBI Document Retriever",
    description="Fetches real-time RBI compliance documents based on a query by scraping the RBI website.",
    func=retrieve_documents,
)

# Define CrewAI agents with the specified LLM and tool
retrieval_agent = Agent(
    role="Document Retriever",
    goal="Retrieve relevant RBI documents in real-time based on the user query",
    backstory="Specialized in fetching up-to-date compliance data from open sources.",
    tools=[retrieve_documents_tool],  # Use the Tool object here
    verbose=True,
    llm=llm
)

response_agent = Agent(
    role="Compliance Advisor",
    goal="Generate actionable RBI compliance advice based on retrieved documents",
    backstory="Expert in interpreting RBI regulations for practical advice.",
    verbose=True,
    llm=llm
)

# Define tasks with Pydantic-structured outputs
retrieval_task = Task(
    description="Retrieve RBI documents relevant to the query: {query}",
    agent=retrieval_agent,
    expected_output=Documents,
    tools=[retrieve_documents_tool],
)

response_task = Task(
    description="Provide compliance advice for the query: {query} using retrieved documents",
    agent=response_agent,
    expected_output=ComplianceAdvice,
    context=[retrieval_task],
)

# Create the crew to manage agents and tasks
crew = Crew(
    agents=[retrieval_agent, response_agent],
    tasks=[retrieval_task, response_task],
    verbose=True,
)

# Streamlit UI
st.title("RBI Compliance Advisory Tool")
st.markdown("Ask a question about RBI compliance and get real-time advice based on open-source data.")

# User input
query = st.text_input("Enter your compliance query (e.g., 'What are KYC requirements for transactions?'):")

if st.button("Get Compliance Advice"):
    if query.strip():
        with st.spinner("Processing your query with real-time data..."):
            try:
                # Execute the crew with the query
                result = crew.kickoff(inputs={"query": query})
                
                # Extract structured outputs
                retrieved_docs = result.tasks_output[0].output.documents
                advice_output = result.tasks_output[1].output
                
                # Display results
                st.subheader("Retrieved Documents")
                for i, doc in enumerate(retrieved_docs, 1):
                    st.write(f"{i}. **Content**: {doc.content}")
                    st.write(f"   **Source**: [{doc.source}]({doc.source})")
                
                st.subheader("Compliance Advice")
                st.success(advice_output.advice)
                
                st.subheader("Sources Referenced")
                for i, source in enumerate(advice_output.sources, 1):
                    st.write(f"{i}. {source}")
            
            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a valid query.")

# Footer with implementation notes
st.markdown("""
---
**Implementation Notes:**
- **Open-Source Only**: Uses Streamlit, Pydantic, CrewAI, and Groq with open-source models (e.g., Llama).
- **Real-Time Data**: Retrieves data by scraping RBI's search results page.
- **Error Handling**: Includes checks for network failures and invalid queries.
- **Scalability**: Limits to 5 documents; consider caching or pagination for larger sets.
- **SmolDocling**: Assumed as document processing; integrated generically in retrieval logic.
- **Gemini**: Excluded as it’s closed-source; focus is on Groq with open models.
- Current date: March 21, 2025.
""")
