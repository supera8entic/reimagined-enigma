# utils.py
def calculate_relevance_score(content: str, query: str) -> float:
    """Calculate document relevance score based on query match."""
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Simple relevance scoring based on query presence
    score = 0.0
    if query_lower in content_lower:
        score += 0.5
    if any(word in content_lower for word in query_lower.split()):
        score += 0.3
    if len(content) > 100:
        score += 0.2
        
    return min(score, 1.0)

def display_results(result):
    """Display results with improved formatting and interactivity."""
    st.subheader("Retrieved Documents")
    retrieved_docs = result.tasks_output[0].output.documents
    for i, doc in enumerate(retrieved_docs, 1):
        with st.expander(f"Document {i}"):
            st.write(f"**Content**: {doc.content}")
            st.write(f"**Source**: [{doc.source}]({doc.source})")
            st.write(f"**Relevance Score**: {doc.relevance_score:.2f}")
            st.write(f"**Last Updated**: {doc.last_updated}")
    
    st.subheader("Compliance Advice")
    advice_output = result.tasks_output[1].output
    st.success(advice_output.advice)
    
    st.subheader("Sources Referenced")
    for i, source in enumerate(advice_output.sources, 1):
        st.write(f"{i}. {source}")

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query with detailed feedback."""
    if not query.strip():
        return False, "Please enter a valid query"
    if len(query) < 10:
        return False, "Query should be at least 10 characters long"
    return True, ""
