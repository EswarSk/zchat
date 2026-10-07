from qdrant_client import AsyncQdrantClient, models


class QdrantStore:
    def __init__(self, client: AsyncQdrantClient, settings):
        self.client = client
        self.collection = settings.qdrant_collection
        self.batch_size = settings.embedding_batch_size

    async def initialize(self, dimension: int):
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                self.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        info = await self.client.get_collection(self.collection)
        dense = info.config.params.vectors
        sparse = info.config.params.sparse_vectors
        if not isinstance(dense, dict) or "dense" not in dense or dense["dense"].size != dimension:
            raise ValueError("Existing Qdrant collection has incompatible dense schema")
        if not sparse or "sparse" not in sparse or sparse["sparse"].modifier != models.Modifier.IDF:
            raise ValueError("Existing Qdrant collection has incompatible sparse schema")
        for field in ["chat_id", "document_id", "node_id", "parent_id", "node_type"]:
            await self.client.create_payload_index(
                self.collection,
                field,
                models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    @staticmethod
    def scope(chat_id: str, document_ids: list[str]):
        if not chat_id or not document_ids:
            raise ValueError("Retrieval requires chat and READY document IDs")
        return models.Filter(
            must=[
                models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id)),
                models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)),
            ]
        )

    async def upsert(self, chat_id, document, nodes, dense, sparse):
        if any(n.chat_id != chat_id or n.document_id != document.id for n in nodes):
            raise ValueError("Index provenance mismatch")
        for start in range(0, len(nodes), self.batch_size):
            points = []
            for node, dense_vector, sparse_vector in zip(
                nodes[start : start + self.batch_size],
                dense[start : start + self.batch_size],
                sparse[start : start + self.batch_size],
                strict=True,
            ):
                payload = node.model_dump(exclude={"text", "metadata", "level", "heading"})
                payload.update(
                    node_id=node.id,
                    filename=document.filename,
                    raw_text=node.raw_text,
                    retrieval_text=node.retrieval_text,
                )
                points.append(
                    models.PointStruct(
                        id=node.id,
                        vector={
                            "dense": dense_vector,
                            "sparse": models.SparseVector(
                                indices=sparse_vector.indices.tolist(),
                                values=sparse_vector.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )
                )
            await self.client.upsert(self.collection, points=points, wait=True)

    async def search(self, chat_id, document_ids, vector, kind, limit):
        if not document_ids:
            return []
        if kind == "sparse":
            vector = models.SparseVector(
                indices=vector.indices.tolist(), values=vector.values.tolist()
            )
        result = await self.client.query_points(
            self.collection,
            query=vector,
            using=kind,
            limit=limit,
            query_filter=self.scope(chat_id, document_ids),
            with_payload=False,
        )
        return [(str(point.id), point.score) for point in result.points]

    async def delete_document(self, chat_id, document_id):
        await self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=self.scope(chat_id, [document_id]),
            ),
            wait=True,
        )

    async def ping(self):
        await self.client.get_collection(self.collection)

    async def close(self):
        await self.client.close()
