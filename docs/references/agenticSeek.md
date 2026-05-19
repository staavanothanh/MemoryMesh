# agenticSeek — Manus AI Alternative (Local)

**Repository:** [https://github.com/Fosowl/agenticSeek](https://github.com/Fosowl/agenticSeek)
**Mức độ liên quan đến MemoryMesh:** ⭐⭐ (Thấp, chỉ tham khảo gián tiếp)

## Tổng quan

agenticSeek là một hệ thống multi-agent chạy hoàn toàn local, hoạt động như một trợ lý tự động có thể duyệt web, viết code, và thực hiện các tác vụ phức tạp. Nó tập trung vào năng lực tự chủ (autonomous agent) hơn là quản lý trí nhớ.

## Tính năng nổi bật

### 1. Multi-Agent Collaboration
- Nhiều agent chuyên biệt cộng tác để giải quyết một nhiệm vụ.
- Có cơ chế phân chia công việc và tổng hợp kết quả.
→ *Áp dụng:* Nếu sau này MemoryMesh mở rộng thành hệ thống multi-agent (ví dụ: một agent chuyên nhớ, một agent chuyên suy luận), có thể tham khảo cách phối hợp.

### 2. Privacy-First, All Local
- Tất cả hoạt động đều diễn ra trên máy người dùng, không gửi dữ liệu ra ngoài.
- Tương thích với các model local như Ollama.
→ *Áp dụng:* Củng cố triết lý "local-first" của MemoryMesh: dữ liệu ký ức của người dùng không bao giờ rời máy.

### 3. Tích hợp DeepSeek qua Ollama
- Có cấu hình sẵn để dùng model DeepSeek local.
→ *Áp dụng:* Là một ví dụ về cách giao tiếp với DeepSeek model trong môi trường local (có thể dùng làm reference cho router_client.py khi kết nối với 9Router hoặc trực tiếp).

## Hạn chế đối với MemoryMesh

- Không tập trung vào memory management.
- Không có cơ chế lưu trữ và truy xuất ký ức người dùng qua các phiên.
- Kiến trúc agent phức tạp, quá mức cần thiết cho giai đoạn đầu của MemoryMesh.

## Kết luận

Repo này chỉ nên được tham khảo khi MemoryMesh đã ổn định và có nhu cầu mở rộng sang hướng autonomous agent. Trong giai đoạn 1-5, ta có thể bỏ qua.