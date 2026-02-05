# services.py
import os
import shutil
import pandas as pd
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# ---------------- CONFIG ----------------
csv_path = r"C:\Users\GenAIBLRANCUSR62\Downloads\review\Amazon_Unlocked_Mobile.csv"

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"

chroma_dir = "chroma_amazon_reviews"
ollama_url = "http://localhost:11434"
embed_model = "llama-3.2-3b-it:latest"
chat_model = "llama-3.2-3b-it:latest"

# ---------------- NEO4J SETUP ----------------
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def create_graph(tx, product, brand, rating, review, votes):
    tx.run("""
        MERGE (p:Product {name: $product})
        MERGE (b:Brand {name: $brand})
        MERGE (p)-[:BELONGS_TO]->(b)
        CREATE (r:Review {text: $review, rating: $rating, votes: $votes})
        CREATE (r)-[:REVIEWS]->(p)
    """,
    product=product,
    brand=brand,
    rating=rating,
    review=review,
    votes=votes
    )

def load_csv_to_neo4j():
    df = pd.read_csv(csv_path, low_memory=False).head()
    with driver.session() as session:
        for _, row in df.iterrows():
            if pd.notna(row["brand_name"]) and pd.notna(row["product_name"]):
                session.execute_write(
                    create_graph,
                    row["product_name"],
                    row["brand_name"],
                    row["ratings"],
                    row["reviews"],
                    row["review_votes"]
                )
    print("✅ Data loaded into Neo4j")

def fetch_reviews():
    with driver.session() as session:
        return session.run("""
            MATCH (r:Review)-[:REVIEWS]->(p:Product)-[:BELONGS_TO]->(b:Brand)
            RETURN p.name AS product,
                   b.name AS brand,
                   r.text AS review,
                   r.rating AS rating,
                   r.votes AS votes
        """).data()

# ---------------- LLM SETUP ----------------
llm = ChatOllama(
    model=chat_model,
    base_url=ollama_url,
    temperature=0.3
)

# ---------------- CHROMA VECTOR STORE ----------------
def build_chroma(reviews):
    documents = []
    for row in reviews:
        documents.append(
            Document(
                page_content=f"""
Product: {row['product']}
Brand: {row['brand']}
Rating: {row['rating']}
Votes: {row['votes']}
Review: {row['review']}
""",
                metadata={
                    "brand": row["brand"],
                    "product": row["product"],
                    "rating": row["rating"],
                    "votes": row["votes"]
                }
            )
        )

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print("🧹 Old ChromaDB deleted")

    chroma_store = Chroma.from_documents(
        docs,
        OllamaEmbeddings(model=embed_model, base_url=ollama_url),
        persist_directory=chroma_dir
    )
    print("✅ Chroma vector store ready")
    return chroma_store
