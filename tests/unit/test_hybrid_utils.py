from memorymesh.memory.hybrid_utils import RRFWithWeights


def test_fuse_both_sources():
    fuser = RRFWithWeights(k=60, weight_vec=0.7, weight_fts=0.3)
    vec_results = [
        {"id": "a", "content": "alpha", "score": 0.9},
        {"id": "b", "content": "beta", "score": 0.8},
    ]
    fts_results = [
        {"id": "b", "content": "beta", "score": 0.1},
        {"id": "c", "content": "gamma", "score": 0.2},
    ]

    fused = fuser.fuse(vec_results, fts_results, top_k=2)
    assert len(fused) == 2
    ids = [r["id"] for r in fused]
    assert ids[0] == "b"  # present in both → highest RRF score


def test_fuse_only_vector():
    fuser = RRFWithWeights()
    vec_results = [
        {"id": "a", "content": "alpha"},
        {"id": "b", "content": "beta"},
    ]

    fused = fuser.fuse(vec_results, [], top_k=2)
    assert len(fused) == 2
    assert fused[0]["id"] == "a"


def test_fuse_only_fts():
    fuser = RRFWithWeights()
    fts_results = [
        {"id": "x", "content": "X-ray"},
        {"id": "y", "content": "Yankee"},
    ]

    fused = fuser.fuse([], fts_results, top_k=2)
    assert len(fused) == 2
    assert fused[0]["id"] == "x"


def test_fuse_empty_both():
    fuser = RRFWithWeights()
    fused = fuser.fuse([], [], top_k=5)
    assert fused == []


def test_fuse_top_k_limits():
    fuser = RRFWithWeights()
    vec = [{"id": str(i)} for i in range(10)]
    fts = [{"id": str(i)} for i in range(5, 15)]

    fused = fuser.fuse(vec, fts, top_k=3)
    assert len(fused) == 3


def test_fuse_custom_weights():
    fuser = RRFWithWeights(k=10, weight_vec=1.0, weight_fts=0.0)
    vec = [{"id": "a"}, {"id": "b"}]
    fts = [{"id": "b"}, {"id": "a"}]

    fused = fuser.fuse(vec, fts, top_k=2)
    assert fused[0]["id"] == "a"  # weight_vec dominates → vector rank decides
