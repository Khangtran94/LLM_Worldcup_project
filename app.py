import streamlit as st
import duckdb
import os
from agent import query_worldcup_agent

# Set page config
st.set_page_config(
    page_title="World Cup SQL Agent",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("🏆 FIFA World Cup SQL Agent")
st.markdown(
    """
    Ask questions about World Cup history, match results, goals, teams, and venues from 1930 to 2026.
    The agent translates your question into a SQL query, runs it on a **DuckDB database** (populated using `dlt`), 
    and presents the final answer.
    """
)

# Database Stats for Sidebar
@st.cache_data
def get_db_stats():
    db_path = os.path.abspath("worldcup.db")
    if not os.path.exists(db_path):
        return 0, 0, 0
    try:
        conn = duckdb.connect(db_path)
        tournaments = conn.execute("SELECT COUNT(DISTINCT year) FROM worldcup_data.matches").fetchone()[0]
        matches = conn.execute("SELECT COUNT(*) FROM worldcup_data.matches").fetchone()[0]
        goals = conn.execute("SELECT COUNT(*) FROM worldcup_data.goals").fetchone()[0]
        conn.close()
        return tournaments, matches, goals
    except Exception:
        return 0, 0, 0

tournaments, total_matches, total_goals = get_db_stats()

# Sidebar Setup
with st.sidebar:
    st.header("📊 Database Statistics")
    st.metric(label="Tournaments Ingested", value=tournaments)
    st.metric(label="Total Matches", value=total_matches)
    st.metric(label="Total Goals Scored", value=total_goals)
    
    st.markdown("---")
    st.header("💡 Try these questions:")
    sample_questions = [
        "Who scored the final goal of the 2022 World Cup, and in what minute?",
        "List all matches played by England in the 2018 World Cup and their scores.",
        "How many goals did Miroslav Klose score in total?",
        "What was the highest scoring match in World Cup history (most goals)?",
        "Who scored the first goal in World Cup history (1930), and what minute was it?",
        "List the top 5 goal scorers in World Cup history and how many goals they scored."
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.user_query = q

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# If user clicked a sample question, set it
if "user_query" in st.session_state and st.session_state.user_query:
    user_input = st.session_state.user_query
    # Reset it so it doesn't trigger on rerun
    st.session_state.user_query = None
else:
    user_input = st.chat_input("Ask a question about the World Cup...")

# Render historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("🔍 Inspect Agent Thought Process & SQL"):
                st.code(msg["sql"], language="sql")
                if "results" in msg and msg["results"]:
                    st.write("Database Results:")
                    st.dataframe(msg["results"])

# Handle new user input
if user_input:
    # Render user query
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Render assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning and querying database..."):
            res = query_worldcup_agent(user_input)
            
            # Extract values
            answer = res.get("answer", "No answer generated.")
            sql_query = res.get("sql_query", "")
            results = res.get("results", [])
            thought = res.get("thought", "")
            error = res.get("error", None)
            
            st.markdown(answer)
            
            # Show debug elements
            if error:
                st.error(f"SQL Error: {error}")
            
            with st.expander("🔍 Inspect Agent Thought Process & SQL"):
                if thought:
                    st.markdown(f"**Thought:** {thought}")
                st.code(sql_query, language="sql")
                if results:
                    st.write("Database Results:")
                    st.dataframe(results)
                    
        # Append message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": sql_query,
            "results": results
        })
