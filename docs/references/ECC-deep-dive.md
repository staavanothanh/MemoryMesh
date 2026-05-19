# ECC Deep Dive – Kỹ thuật áp dụng cho MemoryMesh (DeepSeek-TUI)

**Repository gốc:** [everything-claude-code](https://github.com/affaan-m/everything-claude-code)  
**Mục tiêu:** Trích xuất các kỹ thuật từ ECC và điều chỉnh để MemoryMesh (MCP server) hỗ trợ tối ưu cho DeepSeek & DeepSeek-TUI.

---

## 1. Tự động hóa Memory qua Hooks (PostToolUse, SessionEnd)

**ECC:** Sử dụng hooks của Claude Code (PostToolUse, SessionStart, Stop) để tự động lưu context, cập nhật instincts.  
**Vấn đề:** DeepSeek-TUI chưa có hệ thống hooks phong phú như Claude Code. Tuy nhiên, MemoryMesh là **MCP server** – nó có thể tự ghi nhận mỗi khi tool của mình được gọi (remember, recall) và kết hợp với một vòng lặp nền để mô phỏng hooks.

**Kỹ thuật áp dụng:**
- **PostToolUse hook nội bộ:** Khi bất kỳ MCP tool nào của MemoryMesh được gọi (ví dụ: `remember`), server tự động lưu sự kiện này vào một bảng `audit_log` với thời gian, input, output.  
- **SessionEnd hook qua heartbeat/timeout:** Vì DeepSeek-TUI không báo hiệu kết thúc phiên, MemoryMesh có thể dùng cơ chế timeout (nếu 5 phút không có request) để coi là kết thúc phiên, lúc đó tự động thực hiện:
  - Tóm tắt toàn bộ audit log của phiên thành một ký ức tổng hợp.
  - Cập nhật instinct nếu phát hiện lỗi lặp lại.
  - Giải phóng bộ nhớ tạm.

**Mã giả (Python trong MemoryMesh server):**
```python
import asyncio, time

class MemoryMeshServer:
    def __init__(self):
        self.last_activity = time.time()
        self.session_log = []

    async def tool_remember(self, text):
        # ... xử lý lưu trữ
        self.session_log.append({"tool": "remember", "input": text, "time": time.time()})
        self.last_activity = time.time()
        # Kích hoạt PostToolUse hook bất đồng bộ
        asyncio.create_task(self.post_tool_use_hook())

    async def post_tool_use_hook(self):
        # Cập nhật instinct, hoặc ghi log chi tiết
        pass

    async def check_session_timeout(self):
        while True:
            await asyncio.sleep(60)
            if time.time() - self.last_activity > 300:  # 5 phút
                await self.end_session()

    async def end_session(self):
        # Tóm tắt session log thành memory dài hạn
        # Cập nhật instincts nếu có pattern lỗi
        pass
```
**Lợi ích:** Không phụ thuộc vào client; mọi client MCP (kể cả DeepSeek-TUI) đều được hỗ trợ.

## 2. Continuous Learning Engine: Instincts & Homunculus
**ECC:** Sử dụng một hệ thống rule (instincts) được sinh ra từ quan sát, có confidence score, trạng thái pending → active → superseded. Homunculus (meta-agent) giám sát và đề xuất cải thiện.
**Áp dụng cho MemoryMesh:** MemoryMesh có thể có một module instinct_engine.py chạy ngầm. Thay vì một meta-agent phức tạp, ta có thể dùng chính LLM (qua 9Router) để phân tích log định kỳ và đề xuất instinct mới.
**Quy trình:**
- Observation: MemoryMesh ghi lại các sự kiện bất thường: mâu thuẫn ký ức (cosine similarity cao nhưng nội dung trái ngược), người dùng sửa/xóa memory, hoặc LLM trả lời sai dựa trên memory cũ.
- Pattern Detection: Định kỳ (hoặc khi đủ N sự kiện), gọi LLM (Flash) để phân tích log và đề xuất rule.
- Instinct Generation: Mỗi rule có cấu trúc:
```json
{
  "id": "instinct_001",
  "condition": "when user corrects a memory about a date",
  "action": "increase confidence threshold for date extraction",
  "confidence": 0.7,
  "status": "active",
  "created": "2026-05-18T...",
  "evolve_history": []
}
```
- Evolve: Khi rule được áp dụng thành công, confidence tăng; nếu gây lỗi, confidence giảm hoặc chuyển sang superseded.
**Kỹ thuật triển khai trong MemoryMesh:**
- Lưu instincts vào bảng SQLite riêng.
- Khi recall được gọi, ngoài ký ức, trả về cả instinct liên quan (nếu có) để LLM dùng.
- Có thể tạo MCP tool add_instinct để người dùng chủ động dạy.
**Hỗ trợ DeepSeek-TUI:** Instinct là một dạng “system prompt bổ sung” được tự động đưa vào context của LLM mỗi khi cần. Không yêu cầu thay đổi client.

## 3. Token Budget & Strategic Compaction
**ECC:** Cảnh báo vấn đề 200k → 70k context window bị lấp đầy bởi tool definitions. Đề xuất MCP discipline và strategic compaction (nén khi tool call count vượt ngưỡng).
**Áp dụng cho MemoryMesh:**
- Token Budget: Khi MemoryMesh chuẩn bị gọi LLM (để tóm tắt, extract keywords, hoặc xử lý mâu thuẫn), nó ước lượng token của prompt + ngữ cảnh ký ức. Nếu vượt ngưỡng (ví dụ 70% context window của model), tự động nén các ký ức cũ hoặc ít liên quan trước khi gửi.
- Strategic Compaction: MemoryMesh có thể theo dõi số lượng memory được truy xuất trong một phiên. Khi số lượng > K (ví dụ 5), nó sẽ chủ động tạo bản tóm tắt tổng hợp từ các memory đó, lưu thành một memory mới (summary memory) và giảm bớt các chi tiết.
**Cách thực hiện:**
- Dùng tiktoken hoặc thư viện tương tự để đếm token.
- Trong router_client.py, trước khi gọi LLM, kiểm tra token count. Nếu vượt, gọi hàm compact_memories(memory_list) để sinh tóm tắt.
- Có thể mượn ý tưởng từ ReMe: nén 223,838 tokens → 1,105 tokens bằng cách tóm tắt phân cấp.
**Ví dụ mã giả:**
```python
MAX_PROMPT_TOKENS = 60000  # tùy model

def call_llm_with_memories(prompt, memories):
    full_prompt = build_prompt(prompt, memories)
    if count_tokens(full_prompt) > MAX_PROMPT_TOKENS:
        # Nén memories thành summary
        summary = compact_memories(memories)  # gọi LLM Flash để tóm tắt
        full_prompt = build_prompt(prompt, [summary])
    return call_llm(full_prompt)
```

## 4. MCP Discipline: Đăng ký tool động
**ECC:** 14 MCP servers nhưng chỉ bật một subset theo task hiện tại.
**Áp dụng cho MemoryMesh:** MemoryMesh có thể trở thành “MCP gateway” nhỏ, quản lý một tập các tool và cho phép enable/disable theo ngữ cảnh. Nhưng ở quy mô nhỏ, ta tập trung vào các tool cốt lõi. Tuy nhiên, kỹ thuật này hữu ích nếu sau này MemoryMesh mở rộng ra nhiều tool (forget, tag, export, import, setting). Để tránh làm đầy context của DeepSeek-TUI, MemoryMesh nên:
- Cung cấp một tool meta manage_tools(action="enable"|"disable", tool_name) để người dùng (hoặc chính LLM) bật/tắt tool.
- Mặc định chỉ bật remember, recall. Các tool khác (forget, export) ở trạng thái disabled, chỉ bật khi cần.
**Kỹ thuật:** Sử dụng biến toàn cục trong MCP server để lưu trạng thái tool. Khi client yêu cầu danh sách tool, chỉ trả về những tool đang active.
**Lợi ích:** Giảm noise trong prompt của DeepSeek, tiết kiệm token.

## 5. Model Routing: Flash cho tác vụ nhẹ, Pro cho tác vụ nặng
**ECC:** Dùng model rẻ cho việc đơn giản, model mạnh cho việc phức tạp.
**MemoryMesh đã có kế hoạch này qua 9Router:** Hàm call_llm(prompt, model="deepseek-v4-flash") cho extract/tóm tắt; dùng model="deepseek-v4-pro" cho xử lý mâu thuẫn, sinh instinct.
**Cải tiến:** Có thể tự động chọn model dựa trên loại task:
- Loại 1: extraction, summarization → Flash.
- Loại 2: reasoning, conflict resolution → Pro.
- Loại 3: embedding → dùng local model (sentence-transformers) hoặc embedding API.
Trong `memory_manager.py`, ta có thể định nghĩa hằng số:
```python
MODEL_EXTRACT = "deepseek-v4-flash"
MODEL_REASON = "deepseek-v4-pro"
```
**Hỗ trợ DeepSeek-TUI:** Không cần thay đổi gì; MemoryMesh tự quyết định model khi gọi nội bộ.

## 6. Rust Control Plane → Python TUI Dashboard (tương lai)
**ECC 2.0:** Rust TUI dashboard giám sát session.
**Áp dụng:** Sau khi MemoryMesh ổn định, có thể xây dựng một dashboard đơn giản bằng Python (Textual) hoặc web (FastAPI + Jinja2) để hiển thị:
- Số lượng ký ức, instincts.
- Token saved.
- Các ký ức gần đây.
- Biểu đồ cosine similarity.
Có thể tích hợp trực tiếp vào DeepSeek-TUI bằng cách thêm một panel hoặc một command `memory-mesh status`.

# Tổng kết: Các kỹ thuật ECC cụ thể đưa vào lộ trình MemoryMesh
| Kỹ thuật ECC | Cách áp dụng cho MemoryMesh | Giai đoạn |
| :--- | :--- | :--- |
| Hooks (PostToolUse, SessionEnd) | Tự động ghi log, tổng kết phiên bằng timeout | Giai đoạn 3-4 |
| Instincts Engine | Module học rule từ mâu thuẫn, lưu SQLite, kèm confidence | Giai đoạn 4-5 |
| Token Budget & Strategic Compaction | Đếm token trước LLM call, tự động tóm tắt ký ức | Giai đoạn 3-4 |
| MCP Discipline | Cho phép enable/disable tool động, giảm noise | Giai đoạn 2-3 |
| Model Routing (Flash/Pro) | Dùng Flash cho extract, Pro cho reasoning | Giai đoạn 2 |
| Dashboard (tham khảo) | Xây dựng TUI/web dashboard giám sát | Sau Giai đoạn 5 |
*Lưu ý:* Tất cả các kỹ thuật trên đều được thiết kế để hoạt động với MCP server thuần, không yêu cầu thay đổi client (DeepSeek-TUI). Chỉ cần MemoryMesh là một MCP server đúng chuẩn, các client đều có thể hưởng lợi.