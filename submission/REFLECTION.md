# Reflection — Lab 19

**Tên:** Nguyễn Tiến Huân
**Cohort:** 2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Kết quả Precision@10 trên 50 golden queries cho thấy từng mode có thế mạnh riêng:

- **Exact queries** (chứa từ khóa kỹ thuật chính xác): BM25 thắng (96.7%) vì nó khớp trực tiếp chuỗi ký tự — không cần "hiểu" ngữ nghĩa. Semantic chỉ đạt 88.7% do embedding model (`bge-small-en`) đôi khi trả kết quả liên quan nhưng lệch topic.
- **Paraphrase queries** (diễn đạt lại, không chứa từ gốc): Cả hai mode đều yếu (~24–33%) do model embedding chưa tối ưu cho tiếng Việt. Semantic nhỉnh hơn nhờ bắt được ý nghĩa gần, nhưng chênh lệch nhỏ.
- **Mixed queries** (kết hợp từ chính xác + ý diễn đạt): **Hybrid thắng tuyệt đối (100%)** vì RRF kết hợp cả tín hiệu BM25 lẫn vector — document được cả hai retriever xếp cao sẽ nhận điểm gấp đôi.

**Khi nào KHÔNG dùng hybrid?** Khi corpus nhỏ và query luôn chứa từ khóa chính xác (log search, mã lỗi) → pure BM25 nhanh hơn 10× và đủ tốt. Khi cần latency cực thấp (<5ms P99) trên hệ thống tài nguyên hạn chế, chi phí chạy song song 2 retriever là không đáng.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid search đạt 100% Precision@10 trên mixed queries — cho thấy RRF fusion đơn giản nhưng cực kỳ hiệu quả trong thực tế, đặc biệt với cách người dùng thật hay viết câu truy vấn pha trộn.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: 
