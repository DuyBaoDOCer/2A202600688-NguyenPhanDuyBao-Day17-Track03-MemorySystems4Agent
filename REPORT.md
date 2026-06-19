# Báo cáo Kết quả Benchmark — Day 17, Track 3: Memory Systems for AI Agent

**Thời gian chạy:** 2026-06-19  
**Mode:** LIVE (real LLM)  
**Provider:** custom | **Model:** gemini-3.1-flash-lite  
**Base URL:** https://ai-gateway.antco.ai/v1

---

## 1. Standard Benchmark (`data/conversations.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |            28,180 |                 121,618 |                0.068 |            0.347 |                     0 |           0 |
| Advanced |            23,975 |                  84,463 |                0.950 |            0.938 |                   274 |          64 |

### Phân tích Standard Benchmark

- **Cross-session recall** — Baseline: 0.068 | Advanced: **0.950**  
  → Advanced recall cao hơn **0.88 điểm** nhờ `User.md` lưu facts bền vững qua nhiều thread khác nhau. Baseline chỉ có within-session memory nên quên hoàn toàn khi thread mới bắt đầu.

- **Prompt tokens** — Baseline: 121,618 | Advanced: **84,463**  
  → Advanced tiết kiệm **31% prompt tokens** dù phải mang thêm `User.md` vào system prompt. Lý do: compact memory tóm tắt lịch sử cũ nên context mỗi lượt không tăng tuyến tính như Baseline. Compactions: 64 lần — mỗi lần nén giữ lại chỉ các tin nhắn gần nhất.

- **Memory growth** — Baseline: 0 bytes | Advanced: **274 bytes**  
  → `User.md` phát triển theo thời gian. Rủi ro: facts cũ không tự xóa; nếu user cung cấp thông tin sai hoặc đùa giỡn, có thể ô nhiễm profile. Confidence threshold (mặc định 0.6) giúp lọc các extraction không chắc.

- **Compactions** — Advanced: **64**  
  → Số lần compact cao chứng tỏ ngưỡng token bị vượt thường xuyên, nhất là với LLM live (phản hồi dài hơn offline). Trade-off: compact tiết kiệm prompt load nhưng có thể mất chi tiết trong các turns đã nén.

---

## 2. Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|------------------:|------------------------:|---------------------:|-----------------:|----------------------:|------------:|
| Baseline |             5,822 |                  61,040 |                0.000 |            0.300 |                     0 |           0 |
| Advanced |             6,132 |                  28,021 |                1.000 |            1.000 |                   246 |          28 |

### Phân tích Stress Benchmark

- **Cross-session recall** — Baseline: **0.000** | Advanced: **1.000**  
  → Kết quả phân kỳ hoàn toàn. Facts được giới thiệu ở turns đầu không còn trong context của Baseline khi recall question xuất hiện ở thread khác. Advanced lấy lại chính xác từ `User.md`.

- **Prompt tokens** — Baseline: 61,040 | Advanced: **28,021**  
  → Compact memory giúp Advanced **giảm 54% prompt tokens**.  
  Baseline tăng theo **O(N²)**: mỗi lượt thứ N phải đọc lại N-1 lượt trước.  
  Advanced tăng theo **O(K)**: chỉ giữ K tin nhắn gần nhất + 1 đoạn tóm tắt.  
  Trong hội thoại rất dài (>50 turns), chênh lệch này có thể lên hàng chục nghìn tokens — tiết kiệm đáng kể cả chi phí lẫn độ trễ.

- **Compactions triggered:** **28**  
  → 28 lần compact đã kích hoạt, mỗi lần nén messages cũ thành summary + giữ lại số tin nhắn gần nhất theo `compact_keep_messages`. Rủi ro: nếu `compact_keep_messages` quá nhỏ, chi tiết quan trọng của turns trước có thể bị mất trong tóm tắt.

---

## 3. Bonus — Confidence Threshold

Hàm `extract_profile_updates()` gán điểm confidence cho mỗi extraction trước khi ghi vào `User.md`:

| Mức confidence | Loại tuyên bố |
|----------------|---------------|
| 0.92 – 0.95 | Khai báo tường minh: `'mình tên là X'`, `'đồ uống yêu thích là X'` |
| 0.80 – 0.88 | Sửa lỗi / cập nhật: `'giờ mình đang ở X'`, `'chuyển sang X engineer'` |
| 0.70 – 0.78 | Tuyên bố chung: `'mình ở X'`, `'mình là X'` |
| 0.50          | Ngầm định: `'Python'` xuất hiện trong câu bình thường |

**`CONFIDENCE_THRESHOLD = 0.6`** (mặc định) → lọc bỏ implicit interests (0.50), giữ lại tất cả khai báo rõ ràng. Đặt `0.85` để chỉ giữ explicit + corrections.

- **Vấn đề giải quyết:** Không để câu đùa hoặc ngữ cảnh mơ hồ làm ô nhiễm `User.md`.
- **Cải thiện metric:** Recall tăng vì `User.md` chứa ít false positive hơn.
- **Rủi ro thêm vào:** Threshold cao quá → bỏ sót facts thật sự; cần tune theo dataset.

---

## 4. Kết luận

Bài lab này thể hiện rõ trade-off cốt lõi của Memory Systems:

1. **Baseline không nhớ dài hạn** — recall gần như 0 khi chuyển thread.
2. **Advanced thêm `User.md`** → recall tăng vọt lên 0.95–1.0.
3. **Hội thoại dài làm prompt cost tăng mạnh** ở Baseline (O(N²)), trong khi Advanced giữ ổn định nhờ compact memory.
4. **Compact memory** kéo prompt tokens xuống 31–54% tùy độ dài hội thoại.
5. **Hệ thống mạnh hơn đi kèm phức tạp hơn** — cần confidence threshold, conflict handling, và monitoring `User.md` growth để tránh ô nhiễm dữ liệu dài hạn.
