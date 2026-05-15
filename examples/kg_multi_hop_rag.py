import asyncio
import os

from synapsekit import RAG
from synapsekit.retrieval.kg.backends import NetworkXStore


async def main():
    # Ensure OPENAI_API_KEY is set in environment for this example
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY to run this example.")
        return

    print("Initializing Multi-Hop RAG with Knowledge Graph...")

    # Initialize the in-memory Knowledge Graph store
    graph_store = NetworkXStore()

    # Pass graph_store to RAG to enable hybrid vector + graph retrieval
    rag = RAG(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        graph_store=graph_store,
        retrieval_top_k=3,
    )

    # Add a document with complex relationships (company ownership chains)
    doc_text = (
        "GlobalHoldings LLC acquired a 100% stake in TechVentures Inc in 2022. "
        "TechVentures Inc is the parent company of AI-Solutions GmbH. "
        "AI-Solutions GmbH recently purchased DataCorp, a leading data provider."
    )

    print("Adding document and building knowledge graph...")
    await rag.add_async(doc_text)

    # Query that requires multi-hop reasoning
    # Hop 1: DataCorp -> AI-Solutions GmbH
    # Hop 2: AI-Solutions GmbH -> TechVentures Inc
    # Hop 3: TechVentures Inc -> GlobalHoldings LLC
    query = "Who is the ultimate holding company that owns the parent company of the firm that purchased DataCorp?"

    print(f"\nQuery: {query}")
    answer = await rag.ask(query)

    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
