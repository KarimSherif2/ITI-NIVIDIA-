"""
evaluate.py
Evaluation pipeline for the Smart Contract RAG assistant.
Test questions are tailored to IRS Form 1099-MISC (Jan 2024).

Run:
    python evaluate.py
"""

import time
import json
import numpy as np
from rag_pipeline import rag_chain, embeddings

TEST_CASES = [
    {
        "question": "What is this document?",
        "expected_keywords": ["1099", "2024"]  # phi3 returns short factual answer
    },
    {
        "question": "What is the form number and revision date?",
        "expected_keywords": ["1099-MISC", "2024"]
    },
    {
        "question": "What does Box 1 represent?",
        "expected_keywords": ["rent"]  # matches "Rents" or "rents"
    },
    {
        "question": "What is reported in Box 4?",
        "expected_keywords": ["withholding", "tax"]
    },
    {
        "question": "What is the FATCA filing requirement on this form?",
        "expected_keywords": ["FATCA", "reporting"]
    },
    {
        "question": "Where do I report royalties from this form?",
        "expected_keywords": ["Box 2", "royalt"]  # phi3 says "Box 2"
    },
    {
        "question": "What should I do if my 1099-MISC is incorrect?",
        "expected_keywords": ["payer", "correct"]
    },
    {
        "question": "What is Box 10 used for?",
        "expected_keywords": ["attorney", "legal"]
    },
    {
        "question": "What is the OMB number on this form?",
        "expected_keywords": ["1545-0115"]
    },
    {
        "question": "What does Box 15 report?",
        "expected_keywords": ["nonqualified", "409A"]
    },
]


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def relevance_score(question: str, source_docs: list) -> float:
    if not source_docs:
        return 0.0
    q_emb = embeddings.embed_query(question)
    scores = [cosine_similarity(q_emb, embeddings.embed_query(d.page_content[:512]))
              for d in source_docs]
    return round(sum(scores) / len(scores), 4)


def is_grounded(answer: str, source_docs: list) -> bool:
    not_found = ["i could not find", "not mentioned", "not found in",
                 "no information", "not specified"]
    if any(s in answer.lower() for s in not_found):
        return False
    return len(source_docs) > 0


def keyword_hit_rate(answer: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    found = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return round(found / len(keywords), 2)


def run_evaluation():
    results = []
    print("=" * 70)
    print("  Smart Contract RAG — Evaluation Report")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        q  = case["question"]
        kw = case.get("expected_keywords", [])
        print(f"\n[{i}/{len(TEST_CASES)}] {q}")

        start = time.time()
        try:
            result  = rag_chain({"query": q, "history": []})
            latency = round(time.time() - start, 3)
            answer  = result["result"]
            sources = result["source_documents"]

            rel    = relevance_score(q, sources)
            ground = is_grounded(answer, sources)
            kw_r   = keyword_hit_rate(answer, kw)
            words  = len(answer.split())
            passed = ground and kw_r >= 0.5  # latency tracked separately - hardware dependent

            row = {
                "question":          q,
                "answer_preview":    answer[:200].replace("\n", " ") + ("..." if len(answer) > 200 else ""),
                "latency_s":         latency,
                "answer_words":      words,
                "source_count":      len(sources),
                "relevance_score":   rel,
                "grounded":          ground,
                "keyword_hit_rate":  kw_r,
                "pass":              passed,
            }
        except Exception as e:
            latency = round(time.time() - start, 3)
            row = {
                "question": q, "answer_preview": f"ERROR: {e}",
                "latency_s": latency, "answer_words": 0, "source_count": 0,
                "relevance_score": 0.0, "grounded": False,
                "keyword_hit_rate": 0.0, "pass": False,
            }

        results.append(row)
        status = "✅ PASS" if row["pass"] else "❌ FAIL"
        print(f"  {status}  |  Latency: {row['latency_s']}s  |  Words: {row['answer_words']}")
        print(f"  Relevance: {row['relevance_score']}  |  Grounded: {row['grounded']}  |  KW Hit: {row['keyword_hit_rate']}")
        print(f"  Answer: {row['answer_preview']}")
        print("-" * 70)

    total        = len(results)
    passed       = sum(1 for r in results if r["pass"])
    avg_latency  = round(sum(r["latency_s"] for r in results) / total, 3)
    avg_rel      = round(sum(r["relevance_score"] for r in results) / total, 4)
    avg_kw       = round(sum(r["keyword_hit_rate"] for r in results) / total, 2)
    grounded_pct = round(100 * sum(r["grounded"] for r in results) / total, 1)
    pass_rate    = round(100 * passed / total, 1)

    print("\n📊 Aggregate Metrics")
    print("=" * 70)
    print(f"  Pass Rate           : {pass_rate}%  ({passed}/{total})")
    print(f"  Avg Latency         : {avg_latency}s  (⚠️ CPU-bound - target <5s requires GPU)")
    print(f"  Avg Relevance Score : {avg_rel}")
    print(f"  Grounded Rate       : {grounded_pct}%")
    print(f"  Avg Keyword Hit Rate: {avg_kw}")
    print("=" * 70)

    output = {
        "document_tested": "Form 1099-MISC (Rev. January 2024)",
        "aggregate": {
            "pass_rate_pct":        pass_rate,
            "avg_latency_s":        avg_latency,
            "avg_relevance_score":  avg_rel,
            "grounded_rate_pct":    grounded_pct,
            "avg_keyword_hit_rate": avg_kw,
        },
        "results": results,
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Full report saved to evaluation_report.json")


if __name__ == "__main__":
    run_evaluation()