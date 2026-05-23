"""Milvus Lite collection bootstrap, upsert, and search."""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from ingestion.chunk import Chunk

DEFAULT_COLLECTION = "products"


class MilvusStore:
    def __init__(self, uri: str, dim: int, collection_name: str = DEFAULT_COLLECTION):
        self.uri = uri
        self.dim = dim
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri)

    def ensure_collection(self) -> None:
        if self.client.has_collection(collection_name=self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("product_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("sub_category", DataType.VARCHAR, max_length=64)
        schema.add_field("brand", DataType.VARCHAR, max_length=64)
        schema.add_field("base_price", DataType.FLOAT)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors), "chunk and vector counts must match"
        rows = []
        for chunk, vec in zip(chunks, vectors):
            rows.append({
                "chunk_id": chunk.chunk_id,
                "product_id": chunk.product_id,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "category": chunk.category,
                "sub_category": chunk.sub_category,
                "brand": chunk.brand,
                "base_price": chunk.base_price,
                "embedding": vec,
            })
        # MilvusClient.upsert handles update-or-insert by primary key.
        self.client.upsert(collection_name=self.collection_name, data=rows)

    def search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=["chunk_id", "product_id", "chunk_type", "text",
                          "category", "sub_category", "brand", "base_price"],
        )
        # MilvusClient.search returns [[{entity: {...}, distance: ...}, ...]]
        hits = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append({
                **entity,
                "score": hit.get("distance"),
            })
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return int(stats.get("row_count", 0))
