import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from agent import query_worldcup_agent

# Load environment variables
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class JudgeResponse(BaseModel):
    correctness: bool = Field(description="True if the agent's answer is conceptually correct and accurate, False otherwise")
    explanation: str = Field(description="Brief explanation of the grading decision")

GOLDEN_QUESTIONS = [
    {
        "id": 1,
        "question": "Who won the final match of the 2018 World Cup, and what was the score?",
        "ground_truth": "France won against Croatia, with a score of 4-2.",
        "key_phrases": ["France", "Croatia", "4-2"]
    },
    {
        "id": 2,
        "question": "How many goals did Miroslav Klose score in total across all World Cups?",
        "ground_truth": "Miroslav Klose scored 16 goals in total.",
        "key_phrases": ["16"]
    },
    {
        "id": 3,
        "question": "What was the highest scoring match in World Cup history (most total goals in a single match)? List the teams and the score.",
        "ground_truth": "Austria vs Switzerland in 1954, which ended 7-5 (12 goals total).",
        "key_phrases": ["Austria", "Switzerland", "7-5", "12"]
    },
    {
        "id": 4,
        "question": "How many matches ended in a draw (tie) in the 2014 World Cup?",
        "ground_truth": "9 matches ended in a draw (equal goals for team1 and team2) in the 2014 World Cup.",
        "key_phrases": ["9"]
    },
    {
        "id": 5,
        "question": "Who scored the first goal in World Cup history (1930), what team did they play for, and what minute was it?",
        "ground_truth": "Lucien Laurent (L. Laurent) scored the first goal for France in the 19th minute on July 13, 1930.",
        "key_phrases": ["Laurent", "France", "19"]
    }
]

JUDGE_SYSTEM_PROMPT = """You are an independent evaluator grading the accuracy of an LLM agent answering questions about the FIFA World Cup database.
You will be provided with:
1. The User Question
2. The Ground Truth Answer
3. The Agent's Answer

Your task is to determine if the Agent's Answer is conceptually correct and contains the key facts specified in the Ground Truth.
Output a JSON object with 'correctness' (boolean) and 'explanation' (string).
"""

def judge_answer(question: str, ground_truth: str, agent_answer: str):
    prompt = f"""User Question: {question}
Ground Truth: {ground_truth}
Agent's Answer: {agent_answer}
"""
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format=JudgeResponse
        )
        parsed = response.choices[0].message.parsed
        return parsed.correctness, parsed.explanation
    except Exception as e:
        return False, f"Failed to run LLM judge: {e}"

def run_evaluation():
    print("Starting evaluation of World Cup SQL Agent...")
    print("-" * 50)
    
    total = len(GOLDEN_QUESTIONS)
    passed_phrases = 0
    passed_judge = 0
    results = []
    
    for item in GOLDEN_QUESTIONS:
        q_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        key_phrases = item["key_phrases"]
        
        print(f"\nRunning Case {q_id}: '{question}'")
        
        # Run agent
        agent_res = query_worldcup_agent(question)
        agent_answer = agent_res.get("answer", "")
        sql_query = agent_res.get("sql_query", "")
        
        print(f"SQL: {sql_query}")
        print(f"Answer: {agent_answer}")
        
        # Programmatic check
        phrase_check = all(phrase.lower() in agent_answer.lower() for phrase in key_phrases)
        if phrase_check:
            passed_phrases += 1
            
        # LLM judge check
        correct, explanation = judge_answer(question, ground_truth, agent_answer)
        if correct:
            passed_judge += 1
            
        print(f"Phrase Match: {'PASS' if phrase_check else 'FAIL'} (Expected keywords: {key_phrases})")
        print(f"LLM Judge: {'PASS' if correct else 'FAIL'} - {explanation}")
        
        results.append({
            "id": q_id,
            "question": question,
            "sql": sql_query,
            "answer": agent_answer,
            "phrase_match": phrase_check,
            "judge_correct": correct,
            "explanation": explanation
        })
        
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total Cases: {total}")
    print(f"Programmatic Keyword Pass Rate: {passed_phrases}/{total} ({passed_phrases/total*100:.1f}%)")
    print(f"LLM-as-a-Judge Pass Rate: {passed_judge}/{total} ({passed_judge/total*100:.1f}%)")
    print("=" * 50)
    
    # Save evaluation report to file
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed report saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()
