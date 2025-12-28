import arxiv

def search_arxiv_by_keywords(keywords: list[str], max_results=100):
    query = " AND ".join([f"all:{k}" for k in keywords]) 
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for r in search.results():
        papers.append({
            "title": r.title.strip(),
            "summary": r.summary.strip(),
            "url": r.entry_id,
            "updated": r.updated
        })
    # TODO: 
    return papers
