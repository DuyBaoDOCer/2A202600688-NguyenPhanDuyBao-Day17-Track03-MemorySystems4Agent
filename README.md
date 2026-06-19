# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn (offline/heuristic), `Advanced` có thể tốn hơn `Baseline` về prompt tokens vì overhead User.md chưa được bù bởi compaction
- ở hội thoại dài hoặc với live LLM, compact memory giúp `Advanced` giảm đáng kể prompt tokens (O(K) thay vì O(N²)) và recall tăng rõ rệt nhờ User.md

## Bonus — Confidence Threshold

`extract_profile_updates()` gán điểm confidence cho mỗi fact được trích xuất:

| Khoảng | Ý nghĩa | Ví dụ |
|--------|---------|-------|
| 0.92–0.95 | Khai báo tường minh | "mình tên là Bảo", "đồ uống yêu thích là cà phê" |
| 0.80–0.88 | Sửa lỗi / cập nhật | "giờ mình đang ở Huế", "chuyển sang ML engineer" |
| 0.70–0.78 | Tuyên bố chung | "mình ở Đà Nẵng", "mình là developer" |
| 0.50 | Ngầm định | "Python" xuất hiện trong câu bình thường |

Cấu hình qua `CONFIDENCE_THRESHOLD` (mặc định `0.6`):
- `0.6`: lọc implicit interests, giữ tất cả explicit — cân bằng recall/precision
- `0.85`: strict — chỉ explicit declarations và corrections
- `0.0`: không lọc — dùng cho test / offline

**Vấn đề giải quyết**: ngăn câu đùa và ngữ cảnh mơ hồ làm ô nhiễm User.md  
**Cải thiện**: recall tăng vì User.md chứa ít false positive hơn  
**Rủi ro**: threshold quá cao → bỏ sót facts thật sự; cần tune theo dataset

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.
