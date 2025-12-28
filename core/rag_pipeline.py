from llm.ollama_client import query_ollama
from llm.prompt_templates import rag_answer_prompt

def generate_recommendation(user_question: str, topk_blocks):
    prompt = rag_answer_prompt(user_question, topk_blocks)
    return query_ollama(prompt)
