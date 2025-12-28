def keyword_prompt(user_input: str) -> str:
    return f"""
Extract 3–6 precise keywords for an arXiv query from the user's request.
Only output keywords separated by commas. No extra text.

User request:
\"\"\"{user_input}\"\"\""""

def rag_answer_prompt(user_question: str, topk_blocks: list[tuple[str, dict, float]]) -> str:
    ctx = ""
    for i, (text, meta, score) in enumerate(topk_blocks, 1):
        ctx += f"[{i}] {meta['title']} (score={score:.3f}, updated={meta['updated']})\n{text}\nLink: {meta['url']}\n\n"
    return f"""
You are a research assistant.

User question:
\"\"\"{user_question}\"\"\"

You are given snippets (titles + abstracts) of the most relevant papers. Use ONLY this context as ground truth.
Cite items using [number] and include the links.

Context:
{ctx}

Task: Recommend the best 5 papers for the user's question.
Explain briefly why each was chosen, and include citations like [2], [4].
Return markdown with a numbered list."""
