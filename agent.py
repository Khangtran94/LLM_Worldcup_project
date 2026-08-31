import os
import duckdb
from openai import OpenAI
from pydantic import BaseModel, Field

# Load .env variables manually to avoid python-dotenv dependency
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment or .env file")

client = OpenAI(api_key=api_key)

# Connect to DuckDB
DB_PATH = os.path.abspath("worldcup.db")

class SQLResponse(BaseModel):
    thought: str = Field(description="Chain-of-thought explanation for formulating the SQL query")
    sql: str = Field(description="The executable DuckDB SQL query")

class AnswerResponse(BaseModel):
    thought: str = Field(description="Reflecting on how to formulate the answer based on SQL data")
    answer: str = Field(description="The final natural language response to the user")

SYSTEM_PROMPT_SQL = """You are a precise SQL generator for a DuckDB database containing FIFA World Cup historical data.
Your goal is to write a single DuckDB SQL query to retrieve data that answers the user's question.

The database contains two tables under the schema 'worldcup_data':

1. 'worldcup_data.matches':
- match_id (VARCHAR): Unique identifier for the match.
- tournament (VARCHAR): e.g., 'World Cup 2022'
- year (BIGINT): e.g., 2022
- round (VARCHAR): e.g., 'Matchday 1', 'Semi-finals', 'Final'
- date (VARCHAR): 'YYYY-MM-DD'
- time (VARCHAR): 'HH:MM'
- team1 (VARCHAR): Name of home team (e.g. 'Argentina')
- team2 (VARCHAR): Name of away team (e.g. 'France')
- score_team1 (BIGINT): Full-time goals for team1
- score_team2 (BIGINT): Full-time goals for team2
- score_ht_team1 (BIGINT): Half-time goals for team1
- score_ht_team2 (BIGINT): Half-time goals for team2
- group (VARCHAR): Group name (e.g., 'Group A')
- ground (VARCHAR): Venue / Stadium name and city

2. 'worldcup_data.goals':
- match_id (VARCHAR): Links to the match.
- tournament (VARCHAR): e.g., 'World Cup 2022'
- year (BIGINT): e.g., 2022
- scorer (VARCHAR): Scorer's name (e.g., 'Lionel Messi')
- minute (VARCHAR): Minute of goal (e.g., '34', '90+4')
- penalty (BOOLEAN): True if penalty during match (not shootout)
- owngoal (BOOLEAN): True if own goal
- team (VARCHAR): The team that benefited from the goal (e.g. 'Argentina')
- opponent (VARCHAR): The team that conceded the goal (e.g. 'France')

GUIDELINES:
1. Always qualify tables with the schema 'worldcup_data' (e.g. 'worldcup_data.matches').
2. When searching for team names or player names, use ILIKE with wildcards (e.g. `scorer ILIKE '%Messi%'` or `team1 ILIKE '%France%'`) to handle case-insensitivity, accents, and different spellings.
3. Be careful with own goals: `owngoal` is a boolean.
4. If the user asks for the winner of a match, compare `score_team1` and `score_team2`.
5. The 'minute' column is stored as VARCHAR because it can contain extra time/stoppage time additions (e.g., '90+3', '45+2', '118'). To sort goals chronologically by minute (e.g., to find the last or first goal), do not sort by the raw 'minute' column directly (doing so will sort alphabetically, e.g. '81' > '118'). Instead, sort by parsing the base minute and stoppage minute, for example:
   `ORDER BY CAST(SPLIT_PART(minute, '+', 1) AS INTEGER) DESC, CASE WHEN minute LIKE '%+%' THEN CAST(SPLIT_PART(minute, '+', 2) AS INTEGER) ELSE 0 END DESC` (for descending/latest goals).
6. When asked for the score of a match, always SELECT the individual team scores (`score_team1` and `score_team2`) in your SQL query, not just the total or winner.
7. If the user asks for the first/earliest or last/latest goal of a tournament or across tournaments, you must JOIN the `matches` table to sort by the match `date` (and `time` if available) first, before sorting by goal `minute`.
8. DuckDB SQL syntax is similar to PostgreSQL.
9. Provide ONLY the executable SQL query in the 'sql' field, and your reasoning in the 'thought' field. Do not include markdown code block notation in the 'sql' field itself.
"""

SYSTEM_PROMPT_ANSWER = """You are a helpful football assistant answering questions about FIFA World Cups using data from database queries.
Your task is to take the user's question, the SQL query that was run, and the results returned from the database, and write a helpful, natural language response.

Include details from the query results. If no data was returned, explain that clearly and suggest what might have happened (e.g., no matching tournament/player, incorrect name spelling).
"""

def execute_sql(sql: str):
    conn = duckdb.connect(DB_PATH)
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        # Convert rows to dict list for easier formatting
        results = [dict(zip(columns, row)) for row in rows]
        return results, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def query_worldcup_agent(question: str, model="gpt-4o-mini", max_retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SQL},
        {"role": "user", "content": question}
    ]
    
    attempts = 0
    sql_query = None
    sql_error = None
    results = None
    
    while attempts < max_retries:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=SQLResponse
            )
            parsed_res = response.choices[0].message.parsed
            sql_query = parsed_res.sql.strip()
            thought = parsed_res.thought
            
            # Execute SQL
            results, sql_error = execute_sql(sql_query)
            if not sql_error:
                # Successfully ran!
                break
                
            # If error, append error context and retry
            print(f"Attempt {attempts + 1} SQL failed: {sql_error}\nQuery was: {sql_query}")
            messages.append({"role": "assistant", "content": f"Thought: {thought}\nSQL: {sql_query}"})
            messages.append({
                "role": "user", 
                "content": f"The query failed with error: {sql_error}. Please correct the SQL query and try again."
            })
            attempts += 1
        except Exception as e:
            print(f"Exception during LLM/DB loop: {e}")
            attempts += 1
            
    if sql_error:
        return {
            "question": question,
            "sql_query": sql_query,
            "error": sql_error,
            "answer": f"I attempted to run a database query to answer your question, but encountered an error: {sql_error}."
        }
        
    # Generate natural language answer
    answer_prompt = f"""User Question: {question}
SQL Query Run: {sql_query}
Database Results: {results}
"""
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                {"role": "user", "content": answer_prompt}
            ],
            response_format=AnswerResponse
        )
        parsed_res = response.choices[0].message.parsed
        return {
            "question": question,
            "sql_query": sql_query,
            "results": results,
            "thought": parsed_res.thought,
            "answer": parsed_res.answer
        }
    except Exception as e:
        return {
            "question": question,
            "sql_query": sql_query,
            "results": results,
            "error": str(e),
            "answer": f"Query returned results: {results}, but I couldn't formulate a natural response: {e}"
        }

if __name__ == "__main__":
    # Test query
    test_question = "Who scored the final goal of the 2022 World Cup, and in what minute?"
    print(f"Testing question: '{test_question}'")
    res = query_worldcup_agent(test_question)
    print("\nThought Process:")
    print(res.get("thought"))
    print("\nSQL Generated:")
    print(res.get("sql_query"))
    print("\nResults:")
    print(res.get("results"))
    print("\nAnswer:")
    print(res.get("answer"))
