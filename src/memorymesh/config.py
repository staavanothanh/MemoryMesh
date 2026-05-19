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