import streamlit as st
from services import fetch_reviews, build_chroma, driver
from agents import (
    RetrieverAgent,
    RankerAgent,
    LLMAgent,
    CriticAgent,
    RecommendationAgent,
    PlannerAgent,
    llm
)

# ---------------- INIT ONCE ----------------
@st.cache_resource
def init_system():
    reviews = fetch_reviews()
    chroma_store = build_chroma(reviews)

    retriever = RetrieverAgent(chroma_store)
    ranker = RankerAgent()
    llm_agent = LLMAgent(llm)
    critic = CriticAgent()
    planner = PlannerAgent()

    recommender = RecommendationAgent(
        retriever, ranker, llm_agent, critic
    )

    return planner, recommender

planner, recommender = init_system()

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Agentic Assistant", layout="centered")

st.title("🧠 RAG + Agentic AI Assistant")
st.caption("Ask questions about mobile phones using reviews + knowledge graph")

query = st.text_input(
    "Enter your question",
    placeholder="e.g. Recommend a good Samsung phone under budget"
)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            qtype = planner.classify_query(query)

            if qtype in ["recommendation", "semantic"]:
                answer = recommender.run(query)
                st.success("Answer")
                st.write(answer)

            elif qtype == "count":
                with driver.session() as session:
                    total = session.run(
                        "MATCH (b:Brand) RETURN count(DISTINCT b) AS cnt"
                    ).single()["cnt"]
                st.write(f"Total brands: {total}")

            elif qtype == "list":
                with driver.session() as session:
                    brands = [
                        r["brand"] for r in session.run(
                            "MATCH (b:Brand) RETURN DISTINCT b.name AS brand"
                        )
                    ]
                st.write(brands)

            elif qtype == "aggregation":
                with driver.session() as session:
                    results = session.run("""
                        MATCH (r:Review)-[:REVIEWS]->(:Product)-[:BELONGS_TO]->(b:Brand)
                        RETURN b.name AS brand, round(avg(r.rating),2) AS avg_rating
                        ORDER BY avg_rating DESC
                    """).data()
                st.table(results)
