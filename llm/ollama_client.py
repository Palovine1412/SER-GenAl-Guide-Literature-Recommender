import requests, json

def query_ollama(prompt: str, model="llama3") -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True
    )
    out = ""
    for line in r.iter_lines():
        if line:
            try:
                out += json.loads(line.decode("utf-8")).get("response", "")
            except:
                pass
    return out.strip()
