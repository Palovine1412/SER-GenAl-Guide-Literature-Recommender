from llm.ollama_client import query_ollama
from llm.prompt_templates import keyword_prompt

def extract_keywords(user_input: str) -> list[str]:
    prompt = keyword_prompt(user_input)
    text = query_ollama(prompt)
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
    return parts[:6]