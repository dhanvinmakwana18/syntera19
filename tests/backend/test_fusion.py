from retrieval.fusion import reciprocal_rank_fusion

def test_reciprocal_rank_fusion_weights():
    # Mock data
    dense_candidates = [
        {"id": "doc1", "score": 0.9, "payload": {"text": "D1"}},
        {"id": "doc2", "score": 0.8, "payload": {"text": "D2"}}
    ]
    sparse_candidates = [
        {"id": "doc2", "score": 2.5, "payload": {"text": "D2"}},
        {"id": "doc3", "score": 1.5, "payload": {"text": "D3"}}
    ]
    
    # Base: weights=1.0
    res_base = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, limit=5, dense_weight=1.0, sparse_weight=1.0)
    scores_base = {r["id"]: r["rrf_score"] for r in res_base}
    
    # Doc1: dense rank 0 -> 1/61
    # Doc2: dense rank 1 (1/62), sparse rank 0 (1/61) -> 1/62 + 1/61
    # Doc3: sparse rank 1 -> 1/62
    assert abs(scores_base["doc1"] - (1.0/61)) < 1e-6
    assert abs(scores_base["doc2"] - (1.0/62 + 1.0/61)) < 1e-6
    assert abs(scores_base["doc3"] - (1.0/62)) < 1e-6
    
    # Experiment: dense=0.7, sparse=0.3
    res_exp = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60, limit=5, dense_weight=0.7, sparse_weight=0.3)
    scores_exp = {r["id"]: r["rrf_score"] for r in res_exp}
    
    assert abs(scores_exp["doc1"] - (0.7 * (1.0/61))) < 1e-6
    assert abs(scores_exp["doc2"] - (0.7 * (1.0/62) + 0.3 * (1.0/61))) < 1e-6
    assert abs(scores_exp["doc3"] - (0.3 * (1.0/62))) < 1e-6

    # Verify sorting
    assert res_exp[0]["id"] == "doc2"
    assert res_exp[1]["id"] == "doc1"
    assert res_exp[2]["id"] == "doc3"
