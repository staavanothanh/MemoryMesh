import asyncio
import os
import httpx
import chromadb
from sentence_transformers import SentenceTransformer

from memorymesh.config import AppConfig
from memorymesh.logging_ import setup_logging


async def test_router(config: AppConfig):
    print("Testing 9Router connection...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{config.router.url}/chat/completions",
                json={
                    "model": config.router.default_model,
                    "messages": [{"role": "user", "content": "ping"}]
                }
            )
            if response.status_code == 200:
                print("✓ 9Router is reachable and responding.")
            else:
                print(f"✗ 9Router returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Failed to connect to 9Router: {e}")


def test_embedding(config: AppConfig):
    print("Testing embedding model...")
    try:
        model = SentenceTransformer(config.embedding_model)
        vec = model.encode("Xin chào")
        assert vec.shape == (384,), f"Unexpected shape: {vec.shape}"
        print("✓ Embedding model loaded and works. Shape:", vec.shape)
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")


def test_chromadb(config: AppConfig):
    print("Testing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=config.chroma.db_path)
        collection = client.get_or_create_collection("test_collection")
        collection.add(
            documents=["test memory"],
            metadatas=[{"user_id": "Shinn"}],
            ids=["id1"]
        )
        results = collection.query(
            query_texts=["test"],
            where={"user_id": {"$eq": "Shinn"}},
            n_results=1
        )
        assert len(results["documents"][0]) == 1
        client.delete_collection("test_collection")
        print("✓ ChromaDB works with filter.")
    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")


def test_logging():
    print("Testing logging setup...")
    logger = setup_logging("DEBUG")
    logger.debug("Test debug message")
    if os.path.exists("memorymesh.log"):
        print("✓ Logging to file works (memorymesh.log created).")
    else:
        print("✗ Log file not created.")


async def main():
    print("=== MemoryMesh Environment Verification ===")
    config = AppConfig.from_env()
    config.validate()
    print("Configuration loaded and validated.")
    await test_router(config)
    test_embedding(config)
    test_chromadb(config)
    test_logging()
    print("=== Verification complete ===")


if __name__ == "__main__":
    asyncio.run(main())