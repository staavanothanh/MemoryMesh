## 📊 Phân tích điểm mạnh từ hai chuyên gia

| Khía cạnh | Chuyên gia 1 (Vertical Slices) | Chuyên gia 2 (Enterprise Integration) | Áp dụng cho MemoryMesh? |
|-----------|--------------------------------|----------------------------------------|--------------------------|
| **Cấu trúc thư mục** | `src/memorymesh/`, `tests/unit/`, `tests/integration/`, tách biệt rõ ràng | Interfaces, services, utils | ✅ Áp dụng cấu trúc `src/` package, tách `mcp/`, `memory/`, `schemas/` |
| **Config** | Dataclass `AppConfig` có validation | Không đề cập | ✅ Dùng `AppConfig` dataclass, load từ `.env`, có `validate()` |
| **Error handling** | Custom exceptions (`MemoryMeshError`, `RouterError`, `StorageError`) | Không có phân cấp rõ | ✅ Áp dụng ngay `errors.py` với phân cấp rõ ràng |
| **Types/Schema** | `TypedDict` cho MemoryRecord, SearchResult, ToolInput/Output | Protocol `MemoryBackend` | ✅ Kết hợp cả hai: dùng `Protocol` cho backend, `TypedDict` cho response |
| **Embedder** | Singleton + async-safe (cache model) | AsyncLocalEmbedder với `asyncio.to_thread` | ✅ Giữ lại, code của chuyên gia 1 cụ thể hơn |
| **Router client** | Retry + fallback + circuit breaker pattern | Không đề cập chi tiết | ✅ Áp dụng code chuyên gia 1, thêm circuit breaker đơn giản |
| **Memory backend** | ChromaDB wrapper (abstract protocol) | Đề xuất SQLite + sqlite-vec + FTS5 ngay từ đầu | ⚠️ Giữ ChromaDB cho MVP, abstract protocol để dễ swap sau |
| **Pipeline truy xuất** | Không đề cập (để GĐ4) | 6-stage pipeline (expand, retrieve, fuse, dedup, token budget, score) | ⚠️ Quá phức tạp cho MVP, để dành GĐ4 |
| **Instincts/Entity** | Không có trong MVP | Có bảng `instincts`, `entities` trong schema | ⚠️ Để GĐ4 |
| **Audit log** | Không rõ | `session_audit_logs` table | ✅ Thêm collection "audit_logs" đơn giản trong ChromaDB, chuẩn bị cho hooks |
| **Logging** | Structured logging, stderr | RotatingFileHandler, stderr | ✅ Kết hợp: logging ra stderr + file xoay vòng |
| **Testing** | Unit test từ GĐ2, fixtures | Không có chiến lược test | ✅ Áp dụng chiến lược test của chuyên gia 1 |
| **Handlers** | Tách riêng `handlers.py` | Không tách | ✅ Tách handlers riêng để dễ bảo trì |

---

## 🧩 Đề xuất tích hợp vào MemoryMesh v2.2 (Final)

Tôi sẽ không viết lại toàn bộ plan, mà chỉ nêu **những thay đổi cụ thể** so với bản v2.1 trước đó, cùng với **cấu trúc thư mục mới** và **code nền tảng** (config, errors, types) mà chúng ta sẽ dùng ngay từ GĐ1.

### 1. Cấu trúc thư mục mới

```
D:\Learning_Programing\test-project\memory-mesh\
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── src/
│   └── memorymesh/
│       ├── __init__.py
│       ├── main.py                  # Entry point (stdio/sse)
│       ├── config.py                # AppConfig dataclass
│       ├── errors.py                # Custom exceptions
│       ├── types.py                 # TypedDict schemas
│       ├── logging_.py              # Logging setup
│       ├── embedder.py              # Singleton embedder
│       ├── router.py                # RouterClient (9Router)
│       ├── prompts.py               # System prompts
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── backend.py           # Protocol
│       │   ├── chroma_impl.py       # ChromaDB implementation
│       │   └── manager.py           # Business logic
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py            # Server init, transport
│       │   ├── tools.py             # Tool definitions
│       │   └── handlers.py          # Handler functions
│       └── hooks.py                 # PostToolUse (stub for GĐ4)
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_embedder.py
│   │   ├── test_router.py
│   │   ├── test_memory_manager.py
│   │   └── test_chroma_backend.py
│   ├── integration/
│   │   ├── test_mcp_server.py
│   │   └── test_e2e_workflow.py
│   └── fixtures/
├── docs/
│   ├── plan.md
│   ├── api.md
│   └── references/
├── scripts/
│   ├── verify_env.py
│   └── benchmark.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

### 2. Các file nền tảng sẽ được tạo ngay trong GĐ1

#### `config.py` (theo chuyên gia 1, có chỉnh sửa nhẹ)

```python
from dataclasses import dataclass
from typing import Literal
import os
from dotenv import load_dotenv

@dataclass
class RouterConfig:
    url: str = "http://127.0.0.1:20128/v1"
    default_model: str = "deepseek-v4-flash"
    fallback_model: str = "deepseek-v4-pro"
    timeout_s: int = 30
    max_retries: int = 3
    
    def validate(self):
        from urllib.parse import urlparse
        result = urlparse(self.url)
        assert all([result.scheme, result.netloc]), f"Invalid router URL: {self.url}"

@dataclass
class ChromaConfig:
    db_path: str = "./db/chroma"
    
    def validate(self):
        os.makedirs(self.db_path, exist_ok=True)

@dataclass
class AppConfig:
    router: RouterConfig
    chroma: ChromaConfig
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    default_user_id: str = "Shinn"
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_port: int = 8090
    max_memory_length: int = 2000
    log_level: str = "INFO"
    
    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv(".env")
        return AppConfig(
            router=RouterConfig(
                url=os.getenv("ROUTER_URL", "http://127.0.0.1:20128/v1"),
                default_model=os.getenv("DEFAULT_MODEL", "deepseek-v4-flash"),
                fallback_model=os.getenv("FALLBACK_MODEL", "deepseek-v4-pro"),
                timeout_s=int(os.getenv("ROUTER_TIMEOUT", "30")),
                max_retries=int(os.getenv("ROUTER_RETRIES", "3")),
            ),
            chroma=ChromaConfig(
                db_path=os.getenv("CHROMA_DB_PATH", "./db/chroma"),
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            default_user_id=os.getenv("DEFAULT_USER_ID", "Shinn"),
            mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
            mcp_port=int(os.getenv("MCP_PORT", "8090")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def validate(self):
        self.router.validate()
        self.chroma.validate()
```

#### `errors.py`

```python
class MemoryMeshError(Exception):
    pass

class RouterError(MemoryMeshError):
    def __init__(self, model: str, attempt: int, original_error: Exception):
        self.model = model
        self.attempt = attempt
        self.original_error = original_error
        super().__init__(f"RouterError: model={model}, attempts={attempt}, error={original_error}")

class EmbeddingError(MemoryMeshError):
    pass

class StorageError(MemoryMeshError):
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(f"StorageError({operation}): {reason}")

class ValidationError(MemoryMeshError):
    pass
```

#### `types.py`

```python
from typing import TypedDict

class MemoryRecord(TypedDict):
    id: str
    user_id: str
    content: str
    tags: list[str]
    importance: int
    timestamp: str

class SearchResult(TypedDict):
    id: str
    content: str
    score: float
    tags: list[str]
    importance: int
    timestamp: str

class ToolOutput(TypedDict):
    status: str  # "success" or "error"
    data: dict | list | str | None
    error: str | None
```

#### `logging_.py`

```python
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("memorymesh")
    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # stderr handler (safe for MCP)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # file handler with rotation
    file_handler = RotatingFileHandler(
        "memorymesh.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

### 3. Những gì chúng ta **không** đưa vào MVP (giữ cho GĐ4)

- SQLite + sqlite-vec + FTS5 (vẫn dùng ChromaDB)
- Pipeline 6 giai đoạn (giữ search đơn giản)
- Instincts engine, entity graph
- Memory consolidation

### 4. Audit log đơn giản

Trong `chroma_impl.py`, chúng ta sẽ tạo thêm một collection tên `audit_logs` để ghi lại mỗi lần tool được gọi. Cấu trúc đơn giản: `tool_name`, `input`, `output`, `timestamp`. Việc này sẽ giúp ích cho hooks ở GĐ4 mà không tốn công sức.

### 5. Chiến lược testing

Ngay từ GĐ2, chúng ta sẽ viết unit test cho từng module (`embedder`, `router`, `chroma_backend`, `manager`) và integration test cho MCP server. Sử dụng `pytest` và `pytest-asyncio`.

---

## Bắt đầu Giai đoạn 1

Với những điều chỉnh trên, chúng ta đã có một nền tảng vững chắc, vừa đủ cho MVP, vừa mở rộng được sau này. Bây giờ, bạn có muốn tôi bắt đầu viết code cho **Giai đoạn 1** không? Tôi sẽ:

1. Tạo cấu trúc thư mục như trên.
2. Viết các file nền tảng (`config.py`, `errors.py`, `types.py`, `logging_.py`).
3. Tạo script `verify_env.py` để kiểm tra môi trường.
4. Hướng dẫn bạn chạy từng bước.

Hãy cho tôi biết nếu bạn sẵn sàng!