# everything-claude-code (ECC) — Agent Performance Optimization System

**Repository:** [https://github.com/affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code)

**Mức độ liên quan đến MemoryMesh:** ⭐⭐⭐⭐⭐ (Cực kỳ cao)

## Tổng quan

Everything Claude Code (ECC) không phải là một memory layer hay MCP server đơn lẻ, mà là một **hệ thống tối ưu hiệu năng toàn diện cho AI agent** (đặc biệt là Claude Code). Nó cung cấp các cơ chế tự động hóa, học liên tục, quản lý token và điều phối MCP để giữ cho agent hoạt động thông minh, tiết kiệm và ổn định qua các phiên làm việc dài. Triết lý và kỹ thuật trong ECC có thể áp dụng trực tiếp cho MemoryMesh ở tầng điều phối và tự động hóa trí nhớ.

## Đánh giá theo các trụ cột

| Trụ cột | Điểm số | Khả năng áp dụng cho MemoryMesh |
|:---|:---:|:---|
| **Memory Persistence (Hooks)** | 10/10 | Tự động lưu/nạp ngữ cảnh qua SessionStart/Stop hooks |
| **Continuous Learning (Instincts/Homunculus)** | 10/10 | Học pattern từ hành vi, gán confidence score, evolve rules |
| **Token Optimization** | 9/10 | MCP discipline, strategic compaction, token budget |
| **MCP Management** | 9/10 | 14+ MCP configs, chỉ kích hoạt subset cần thiết |
| **Agent Delegation** | 8/10 | Subagent với context giới hạn, model routing theo task |
| **Rust Control Plane (ECC 2.0)** | 8/10 | TUI dashboard, SQLite state store, session orchestration |

## Những gì MemoryMesh có thể học từ ECC

### 1. Cơ chế Memory Persistence qua Hooks

ECC sử dụng hooks (SessionStart, Stop, PostToolUse) để tự động lưu và khôi phục ngữ cảnh giữa các phiên. Agent không cần tự quyết định khi nào "nhớ" – mọi thứ được ghi lại một cách có hệ thống.

- **SessionStart hook**: Tự động nạp context từ session trước, bao gồm các quyết định, bài học (instincts) và trạng thái dự án.
- **Stop hook**: Lưu toàn bộ trạng thái quan trọng trước khi kết thúc.
- **PostToolUse hook**: Ghi nhận kết quả ngay sau mỗi tool call, tạo cơ hội cập nhật memory tức thì.

**Áp dụng vào MemoryMesh**: Thay vì chỉ dựa vào tool `remember` do người dùng hoặc LLM gọi, MemoryMesh có thể tự động kích hoạt `add_memory` ngầm sau mỗi lượt hội thoại quan trọng hoặc khi kết thúc phiên. Điều này đảm bảo không bỏ sót ký ức và giảm gánh nặng cho LLM.

### 2. Continuous Learning với Instincts & Homunculus

Đây là phần đặc sắc nhất của ECC. Nó có một engine học liên tục:

- **Instincts**: Các quy tắc hành vi được rút ra từ kinh nghiệm (ví dụ: “luôn kiểm tra file trước khi sửa”). Instincts có confidence score, có thể được củng cố hoặc suy yếu qua thời gian.
- **Homunculus**: Một agent nhỏ giám sát hành vi của agent chính, phân tích pattern, đưa ra đề xuất cải thiện (meta‑cognition).
- Trạng thái instinct: `pending` → `active` → `superseded` (hoặc `deprecated`).
- Có thể import/export instincts để chia sẻ giữa các agent hoặc backup.

**Áp dụng vào MemoryMesh**: Kết hợp với self-learning loop của sage-memory, MemoryMesh có thể có một **Learning Engine** riêng. Khi phát hiện mâu thuẫn ký ức hoặc lỗi lặp lại, hệ thống sẽ tạo ra một `instinct` (rule) với confidence score. Các instinct này được lưu trong DB và được recall kèm theo ký ức thông thường, giúp LLM tránh sai lầm. Cơ chế evolve (pending → active → superseded) đảm bảo luôn có phiên bản rule tốt nhất.

### 3. Token Optimization & “200k → 70k Problem”

ECC chỉ ra rằng khi có quá nhiều MCP server (mỗi server mang theo danh sách tool dài), context window bị lấp đầy bởi tool definitions, giảm khả năng suy luận thực sự. Giải pháp của ECC:

- **MCP Discipline**: Cấu hình nhiều server nhưng chỉ enable một subset nhỏ phù hợp với task hiện tại.
- **Strategic Compaction**: Khi số lượng tool call vượt ngưỡng, agent được nhắc chủ động nén context (compaction) thay vì đợi đầy window rồi mới xử lý.
- **Token Budget**: Thiết lập ngân sách token cứng cho từng loại nội dung (system prompt, tools, conversation).

**Áp dụng vào MemoryMesh**: MemoryMesh là một MCP server, do đó nó sẽ đóng góp vào “tool definition overhead”. Ta phải giữ cho định nghĩa tool của MemoryMesh cực kỳ gọn gàng (chỉ 3-4 tool chính). Đồng thời, MemoryMesh có thể hỗ trợ tính năng strategic compaction: khi phát hiện context sắp đầy (dựa trên số token ước lượng), nó sẽ chủ động tóm tắt ký ức dài thành dạng ngắn gọn hơn – giống như ReMe nhưng chủ động hơn.

### 4. MCP Management

ECC cung cấp 14+ cấu hình MCP server khác nhau, từ file system, git, đến memory và web search. Tuy nhiên, nó nhấn mạnh rằng không nên bật tất cả cùng lúc. Có script để tự động kích hoạt/tắt MCP server dựa trên task.

**Áp dụng vào MemoryMesh**: MemoryMesh có thể tận dụng ý tưởng này để trở thành “trung tâm điều phối MCP” nếu trong tương lai tích hợp thêm các công cụ khác. Hiện tại, bài học là: chỉ giữ các tool cốt lõi (remember, recall, forget) luôn sẵn sàng; các tool nâng cao có thể được enable động theo ngữ cảnh.

### 5. Agent Delegation & Model Routing

ECC khuyến nghị dùng subagent cho các tác vụ lớn để giữ context chính sạch sẽ. Nó cũng phân biệt model: dùng model rẻ (Flash) cho việc đơn giản, model mạnh (Pro) cho việc phức tạp.

**Áp dụng vào MemoryMesh**: MemoryMesh khi gọi LLM qua 9Router có thể tận dụng model routing: tóm tắt, extract keywords dùng `deepseek-v4-flash` (rẻ, nhanh); xử lý mâu thuẫn hoặc tạo instinct dùng `deepseek-v4-pro` (chất lượng cao). Điều này đã được lên kế hoạch trong router_client của MemoryMesh.

### 6. Rust Control Plane (ECC 2.0)

ECC 2.0 đang được phát triển với một control plane viết bằng Rust, cung cấp:
- **TUI Dashboard**: Giám sát trực quan session, token usage, instincts.
- **SQLite State Store**: Lưu toàn bộ trạng thái agent.
- **Session Orchestrator**: Quản lý nhiều session song song.

**Áp dụng vào MemoryMesh**: Đây là tầm nhìn dài hạn cho MemoryMesh: sau khi MCP server hoạt động ổn định, có thể xây dựng một dashboard nhỏ (có thể bằng Python Textual hoặc Rust) để người dùng xem được “bộ não” của mình: danh sách ký ức, instinct, token đã tiết kiệm, biểu đồ cosine similarity, v.v.

## Kết luận

Everything Claude Code bổ sung một mảnh ghép hoàn hảo cho bức tranh MemoryMesh: **lớp tự động hóa và tối ưu hóa thông minh**. Trong khi mem0 cung cấp engine lưu trữ, sage-memory cung cấp pipeline truy xuất, ReMe cung cấp cơ chế nén, thì ECC cung cấp triết lý về **hooks, học liên tục và quản lý token** – những thứ biến MemoryMesh từ một công cụ bị động thành một hệ thống chủ động, tự cải thiện theo thời gian.

**Khuyến nghị tích hợp**: Ở Giai đoạn 4 (nâng cấp MemoryMesh), thay vì chỉ thêm tính năng quản lý trùng lặp, ta có thể thiết kế hẳn một **Instinct Engine** dựa trên ý tưởng của ECC, kết hợp với self-learning loop của sage-memory. Các hooks tự động (PostToolUse, SessionEnd) sẽ được thêm vào MCP server để việc ghi nhớ diễn ra ngầm, không cần LLM phải gọi tool `remember` một cách thủ công.