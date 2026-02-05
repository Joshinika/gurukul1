# agents.py
from services import chroma_dir, llm, driver
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document

# ---------------- TEXT PROCESSING AGENT ----------------
class TextProcessingAgent:
    def run(self, docs):
        processed_docs = []
        for doc in docs:
            content = doc.page_content.lower().strip()
            content = " ".join(content.split())  # remove extra whitespace
            processed_docs.append(Document(page_content=content, metadata=doc.metadata))
        return processed_docs

# ---------------- RETRIEVER AGENT ----------------
class RetrieverAgent:
    def __init__(self, vector_store):
        self.store = vector_store

    def run(self, query, k=5):
        return self.store.similarity_search(query, k=50)

# ---------------- RANKER AGENT ----------------
class RankerAgent:
    def run(self, docs):
        # Rank by rating first, then votes
        def score(doc):
            rating = doc.metadata.get("rating", 0) or 0
            votes = doc.metadata.get("votes", 0) or 0
            return rating + votes*0.01
        return sorted(docs, key=score, reverse=True)

# ---------------- LLM AGENT ----------------
class LLMAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query, context_docs):
        context = "\n\n".join(d.page_content for d in context_docs)
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nExplain clearly:"
        return self.llm.invoke(prompt).content

# ---------------- CRITIC AGENT ----------------
class CriticAgent:
    def run(self, recommendation):
        if not recommendation or len(recommendation.strip()) < 5:
            return "⚠ Recommendation unclear, please refine."
        return recommendation

# ---------------- RECOMMENDATION AGENT ----------------
class RecommendationAgent:
    def __init__(self, retriever, ranker, llm_agent, critic):
        self.retriever = retriever
        self.ranker = ranker
        self.llm_agent = llm_agent
        self.critic = critic

    def run(self, query):
        docs = self.retriever.run(query)
        ranked_docs = self.ranker.run(docs)
        llm_response = self.llm_agent.run(query, ranked_docs)
        final_response = self.critic.run(llm_response)
        return final_response

# ---------------- PLANNER AGENT ----------------
class PlannerAgent:
    def classify_query(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["how many", "count", "total"]):
            return "count"
        if any(x in q for x in ["list", "which are", "what are", "show all"]):
            return "list"
        if any(x in q for x in ["best", "recommend", "right choice", "suitable"]):
            return "recommendation"
        if any(x in q for x in ["average", "avg", "rating"]):
            return "aggregation"
        return "semantic"
