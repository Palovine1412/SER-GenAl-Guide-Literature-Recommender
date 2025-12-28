from core.key_word_extractor import extract_keywords
from search.arxiv_client import search_arxiv_by_keywords
from core.vector_store import build_index, search_topk
from core.rag_pipeline import generate_recommendation

if __name__ == "__main__":
    q = input("Enter your research topic/question: ").strip()
    if not q:
        print("Empty input."); exit()

    # 1) LLM 提取关键词（可打印出来助教/调试）
    kws = extract_keywords(q)
    print("Keywords:", kws)

    # 2) arXiv 检索（拿大一点的候选集）
    papers = search_arxiv_by_keywords(kws, max_results=100)
    if not papers:
        print("No papers found."); exit()
    print(f"Retrieved {len(papers)} papers from arXiv.")

    # 3) 建立/覆盖向量库（也可以做成增量/持久化）
    build_index(papers)

    # 4) RAG：用原始问题做 Top-K 语义检索
    topk = search_topk(q, k=5)

    # 5) 基于证据的推荐生成（带引用）
    answer = generate_recommendation(q, topk)
    print("\n===== Recommended Papers =====\n")
    print(answer)
