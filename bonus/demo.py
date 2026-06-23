"""Demo script cho HybridMemoryAgent — 5 query minh hoạ.

Chạy: python bonus/demo.py
Yêu cầu: đã chạy setup-lite.sh + feast apply + feast materialize-incremental.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonus.agent import HybridMemoryAgent


def main() -> None:
    print("=" * 70)
    print("  HybridMemoryAgent — Demo 5 Queries")
    print("=" * 70)

    # ── Khởi tạo agent ───────────────────────────────────────────────────
    print("\n[init] Khởi tạo agent (load embedding model + Qdrant + Feast)...")
    agent = HybridMemoryAgent()

    # ── Nạp episodic memories ────────────────────────────────────────────
    print("\n[seed] Nạp episodic memories cho user u_001...\n")

    memories = [
        "Hôm nay tôi đã đọc một bài viết rất hay về Kubernetes auto-scaling. "
        "Bài viết giải thích cách Horizontal Pod Autoscaler hoạt động dựa trên "
        "CPU utilization và custom metrics. Tôi thấy rất hữu ích cho dự án "
        "đang làm ở công ty.",

        "Tôi vừa tìm hiểu về cloud security best practices. Các nguyên tắc "
        "bao gồm: least privilege access, encryption at rest và in transit, "
        "network segmentation, và continuous monitoring. AWS GuardDuty là "
        "công cụ tốt để phát hiện anomaly.",

        "Ghi chú meeting: Team quyết định chuyển từ monolith sang microservices. "
        "Sử dụng Docker + Kubernetes cho orchestration. Timeline: 3 tháng. "
        "Cần training thêm về service mesh (Istio) cho team.",

        "Đã hoàn thành khoá học về machine learning pipeline trên Coursera. "
        "Các bước chính: data ingestion, feature engineering, model training, "
        "evaluation, deployment. MLflow dùng để track experiments.",

        "Tìm hiểu về vector database cho semantic search. Qdrant và Weaviate "
        "là hai lựa chọn phổ biến. Qdrant hỗ trợ filtering tốt, Weaviate "
        "có module tích hợp embedding model. Cả hai đều hỗ trợ hybrid search.",
    ]

    for mem in memories:
        agent.remember(mem, user_id="u_001")

    # ── 5 Queries minh hoạ ───────────────────────────────────────────────
    queries = [
        {
            "desc": "Query 1 — Hỏi đơn giản (chỉ vector hit)",
            "query": "Tôi đã đọc gì về Kubernetes?",
        },
        {
            "desc": "Query 2 — Cần profile context (topic_affinity)",
            "query": "Recommend đọc gì tiếp theo cho tôi?",
        },
        {
            "desc": "Query 3 — Cần fresh activity (queries_last_hour)",
            "query": "Tôi đang quan tâm gì gần đây?",
        },
        {
            "desc": "Query 4 — Paraphrase (vector wins)",
            "query": "Tài liệu về tự động mở rộng hạ tầng theo tải?",
        },
        {
            "desc": "Query 5 — Mixed (hybrid + profile)",
            "query": "Cho tôi summary về cloud security",
        },
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{'─' * 70}")
        print(f"  {q['desc']}")
        print(f"  Query: \"{q['query']}\"")
        print(f"{'─' * 70}")
        context = agent.recall(q["query"], user_id="u_001")
        print(context)

    print(f"\n{'=' * 70}")
    print("  Demo hoàn tất — 5/5 queries thực hiện thành công.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
