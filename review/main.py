# main.py
from services import fetch_reviews, build_chroma, driver
from agents import TextProcessingAgent, RetrieverAgent, RankerAgent, LLMAgent, CriticAgent, RecommendationAgent, PlannerAgent, llm

# ---------------- BUILD VECTOR STORE ----------------
reviews = fetch_reviews()
chroma_store = build_chroma(reviews)

# ---------------- INIT AGENTS ----------------
text_agent = TextProcessingAgent()
retriever = RetrieverAgent(chroma_store)
ranker = RankerAgent()
llm_agent = LLMAgent(llm)
critic = CriticAgent()
planner = PlannerAgent()

recommender = RecommendationAgent(retriever, ranker, llm_agent, critic)

# ---------------- CLI / ORCHESTRATOR ----------------
print("\n💬 Ask questions (type 'exit' to quit)\n")

while True:
    query = input("Query: ").strip()
    if query.lower() == "exit":
        break

    qtype = planner.classify_query(query)

    if qtype == "recommendation" or qtype == "semantic":
        response = recommender.run(query)
        print("\n💡 Recommendation:\n", response)

    elif qtype == "count":
        with driver.session() as session:
            total = session.run("MATCH (b:Brand) RETURN count(DISTINCT b) AS cnt").single()["cnt"]
            print(f"Total brands: {total}")

    elif qtype == "list":
        with driver.session() as session:
            brands = [r["brand"] for r in session.run("MATCH (b:Brand) RETURN DISTINCT b.name AS brand ORDER BY brand")]
            print("Brands:")
            for b in brands:
                print("-", b)

    elif qtype == "aggregation":
        with driver.session() as session:
            results = session.run("""
                MATCH (r:Review)-[:REVIEWS]->(:Product)-[:BELONGS_TO]->(b:Brand)
                RETURN b.name AS brand, round(avg(r.rating),2) AS avg_rating
                ORDER BY avg_rating DESC
            """).data()
            print("Average rating per brand:")
            for r in results:
                print(f"{r['brand']} → {r['avg_rating']} ⭐")

driver.close()
