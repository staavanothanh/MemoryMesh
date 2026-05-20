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
class FTSConfig:
    db_path: str = "./db/memory_fts.db"

    def validate(self):
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)


@dataclass
class ConsolidationConfig:
    similarity_threshold: float = 0.85
    min_cluster_size: int = 2
    interval_seconds: int = 3600
    batch_size: int = 50
    enabled: bool = True

    def validate(self):
        assert 0.0 < self.similarity_threshold <= 1.0, "similarity_threshold must be in (0, 1]"
        assert self.min_cluster_size >= 2, "min_cluster_size must be >= 2"


@dataclass
class AppConfig:
    router: RouterConfig
    chroma: ChromaConfig
    fts: FTSConfig
    consolidation: ConsolidationConfig
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    default_user_id: str = "Shinn"
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_port: int = 8090
    max_memory_length: int = 2000
    log_level: str = "INFO"
    rrf_k: int = 60
    rrf_weight_vec: float = 0.7
    rrf_weight_fts: float = 0.3
    rrf_pool_size: int = 20
    token_budget: int = 1000
    truncation_weight_score: float = 0.6
    truncation_weight_importance: float = 0.3
    truncation_weight_recency: float = 0.1

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
            fts=FTSConfig(
                db_path=os.getenv("FTS_DB_PATH", "./db/memory_fts.db"),
            ),
            consolidation=ConsolidationConfig(
                similarity_threshold=float(os.getenv("CONSOLIDATION_SIMILARITY", "0.85")),
                min_cluster_size=int(os.getenv("CONSOLIDATION_MIN_CLUSTER", "2")),
                interval_seconds=int(os.getenv("CONSOLIDATION_INTERVAL", "3600")),
                batch_size=int(os.getenv("CONSOLIDATION_BATCH", "50")),
                enabled=os.getenv("CONSOLIDATION_ENABLED", "true").lower() == "true",
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            default_user_id=os.getenv("DEFAULT_USER_ID", "Shinn"),
            mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
            mcp_port=int(os.getenv("MCP_PORT", "8090")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            rrf_k=int(os.getenv("RRF_K", "60")),
            rrf_weight_vec=float(os.getenv("RRF_WEIGHT_VEC", "0.7")),
            rrf_weight_fts=float(os.getenv("RRF_WEIGHT_FTS", "0.3")),
            rrf_pool_size=int(os.getenv("RRF_POOL_SIZE", "20")),
            token_budget=int(os.getenv("TOKEN_BUDGET", "1000")),
            truncation_weight_score=float(os.getenv("TRUNCATION_WEIGHT_SCORE", "0.6")),
            truncation_weight_importance=float(os.getenv("TRUNCATION_WEIGHT_IMPORTANCE", "0.3")),
            truncation_weight_recency=float(os.getenv("TRUNCATION_WEIGHT_RECENCY", "0.1")),
        )

    def validate(self):
        self.router.validate()
        self.chroma.validate()
        self.fts.validate()
        self.consolidation.validate()