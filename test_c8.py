import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from app.catalog import catalog_store
from app.schemas import ChatRequest, Message
from app.agent import generate_response

async def test_c8():
    catalog_store.load()
    
    # Simulate C8 conversation
    messages = [
        Message(role="user", content="I need to quickly screen admin assistants for Excel and Word daily."),
        Message(role="assistant", content="For a quick knowledge check, the knowledge-only variants are the right fit. I have excluded simulations due to time constraints. Here are some options: MS Excel (New), MS Word (New), and Occupational Personality Questionnaire OPQ32r."),
        Message(role="user", content="In that case, I am OK with adding a simulation - we want to capture the capabilities.")
    ]
    
    request = ChatRequest(messages=messages)
    
    # check rag
    query = " ".join([m.content for m in messages if m.role == "user"])
    results = catalog_store.search(query, top_k=25)
    print("\n--- RAG Top 25 ---")
    for r in results:
        print(r.name)
        
    response = await generate_response(request)
    
    print("\n--- Agent Reply ---")
    print(response.reply)
    print("\n--- Recommendations ---")
    for rec in response.recommendations:
        print(f"- {rec.name} ({rec.url})")

if __name__ == "__main__":
    asyncio.run(test_c8())
