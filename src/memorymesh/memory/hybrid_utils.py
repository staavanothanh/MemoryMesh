from typing import List, Dict, Any


class RRFWithWeights:
    def __init__(self, k: int = 60, weight_vec: float = 0.7, weight_fts: float = 0.3):
        self.k = k
        self.weight_vec = weight_vec
        self.weight_fts = weight_fts

    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        fts_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        identity_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc["id"]
            identity_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + self.weight_vec * (1.0 / (self.k + rank))

        for rank, doc in enumerate(fts_results, start=1):
            doc_id = doc["id"]
            if doc_id not in identity_map:
                identity_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + self.weight_fts * (1.0 / (self.k + rank))

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        return [identity_map[doc_id] for doc_id in sorted_ids]
