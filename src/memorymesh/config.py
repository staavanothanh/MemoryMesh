from dataclasses import dataclass, field
from typing import Literal, Optional, List
import os
from dotenv import load_dotenv

@dataclass
class RouterConfig:
    url: str = "http://127.0.0.1:20128/v1"
    default_model: str = "deepseek-v4-flash"
    fallback_model: str = "deepseek-v4-pro"
    background_model_pool: Optional[List[str]] = None
    timeout_s: int = 30
    max_retries: int = 3

    def validate(self):
        from urllib.parse import urlparse
        result = urlparse(self.url)
        assert all([result.scheme, result.netloc]), f"Invalid router URL: {self.url}"

@dataclass
class SqliteVecConfig:
    db_path: str = "./db/memory.db"
    auto_migrate: bool = False
    max_concurrent_writes: int = 3

    def validate(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        assert 1 <= self.max_concurrent_writes <= 10, "max_concurrent_writes must be 1–10"


# Deprecated — kept for migration script only
@dataclass
class SessionStartConfig:
    instruction_file_priority: Optional[List[str]] = None
    instructions_dir: str = ".memorymesh/instructions/"
    global_instruction_file: str = ""
    docs_sync_enabled: bool = True
    docs_sync_files: Optional[List[str]] = None
    max_file_size: int = 51200


@dataclass
class SessionConfig:
    db_path: str = "./db/sessions.db"
    auto_create_session: bool = True
    auto_scan_codebase: bool = True
    auto_recall_on_start: bool = True
    auto_compact_on_end: bool = True
    auto_extract_facts: bool = True
    compact_threshold: int = 20
    max_context_log: int = 500
    stale_session_minutes: int = 30
    session_start: SessionStartConfig = field(default_factory=SessionStartConfig)

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
    session_memory_ttl_days: int = 7
    min_importance_to_keep: int = 3
    ephemeral_memory_ttl_days: int = 7

    def validate(self):
        assert 0.0 < self.similarity_threshold <= 1.0, "similarity_threshold must be in (0, 1]"
        assert self.min_cluster_size >= 2, "min_cluster_size must be >= 2"
        assert 1 <= self.min_importance_to_keep <= 5, "min_importance_to_keep must be 1-5"
        assert self.ephemeral_memory_ttl_days >= 0, "ephemeral_memory_ttl_days must be >= 0"


@dataclass
class InstinctConfig:
    db_path: str = "./db/instincts.db"
    enabled: bool = True
    min_samples: int = 5
    v2_max_instincts: int = 200
    max_pattern_length: int = 200
    confidence_floor: float = 0.1
    dedup_similarity_threshold: float = 0.95

    def validate(self):
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        assert 1 <= self.v2_max_instincts <= 10000, "v2_max_instincts must be 1–10000"
        assert self.max_pattern_length >= 10, "max_pattern_length must be >= 10"
        assert 0.0 <= self.confidence_floor <= 1.0, "confidence_floor must be 0.0–1.0"
        assert 0.5 <= self.dedup_similarity_threshold <= 1.0, "dedup_similarity_threshold must be 0.5–1.0"


@dataclass
class EccIntegrationConfig:
    choke_point_enabled: bool = True
    auto_save_tool_context: bool = True
    bootstrap_snapshots: bool = True

    def validate(self):
        pass


@dataclass
class EmbeddingConfig:
    mode: Literal["local", "remote"] = "local"
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    remote_api_url: str = ""
    remote_api_key: str = ""

    def validate(self):
        if self.mode == "remote" and not self.remote_api_url:
            raise ValueError("REMOTE_EMBEDDING_API_URL is required when EMBEDDING_MODE=remote")


@dataclass
class AppConfig:
    router: RouterConfig
    sqlite_vec: SqliteVecConfig
    session: SessionConfig
    consolidation: ConsolidationConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    instinct: InstinctConfig = field(default_factory=InstinctConfig)
    ecc_integration: EccIntegrationConfig = field(default_factory=EccIntegrationConfig)
    level_weight_session: float = 2.0
    level_weight_user: float = 1.5
    level_weight_knowledge: float = 1.0
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    default_user_id: str = "Shinn"
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_port: int = 8090
    log_level: str = "INFO"
    token_budget: int = 1000
    truncation_weight_score: float = 0.6
    truncation_weight_importance: float = 0.3
    truncation_weight_recency: float = 0.1

    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv(".env")

        instruction_files_env = os.getenv("SESSION_INSTRUCTION_FILES")
        docs_files_env = os.getenv("SESSION_DOCS_SYNC_FILES")

        return AppConfig(
            router=RouterConfig(
                url=os.getenv("ROUTER_URL", "http://127.0.0.1:20128/v1"),
                default_model=os.getenv("DEFAULT_MODEL", "deepseek-v4-flash"),
                fallback_model=os.getenv("FALLBACK_MODEL", "deepseek-v4-pro"),
                background_model_pool=os.getenv("BACKGROUND_MODEL_POOL", "").split(",") if os.getenv("BACKGROUND_MODEL_POOL") else None,
                timeout_s=int(os.getenv("ROUTER_TIMEOUT", "30")),
                max_retries=int(os.getenv("ROUTER_RETRIES", "3")),
            ),
            sqlite_vec=SqliteVecConfig(
                db_path=os.getenv("VEC_DB_PATH", "./db/memory.db"),
                auto_migrate=os.getenv("VEC_AUTO_MIGRATE", "false").lower() == "true",
                max_concurrent_writes=int(os.getenv("VEC_MAX_CONCURRENT_WRITES", "3")),
            ),
            session=SessionConfig(
                db_path=os.getenv("SESSION_DB_PATH", "./db/sessions.db"),
                auto_create_session=os.getenv("SESSION_AUTO_CREATE", "true").lower() == "true",
                auto_scan_codebase=os.getenv("SESSION_AUTO_SCAN_CODEBASE", "true").lower() == "true",
                auto_recall_on_start=os.getenv("SESSION_AUTO_RECALL_ON_START", "true").lower() == "true",
                auto_compact_on_end=os.getenv("SESSION_AUTO_COMPACT_ON_END", "true").lower() == "true",
                auto_extract_facts=os.getenv("AUTO_EXTRACT_FACTS", "true").lower() == "true",
                compact_threshold=int(os.getenv("SESSION_COMPACT_THRESHOLD", "20")),
                max_context_log=int(os.getenv("SESSION_MAX_CONTEXT_LOG", "500")),
                stale_session_minutes=int(os.getenv("SESSION_STALE_MINUTES", "30")),
                session_start=SessionStartConfig(
                    instruction_file_priority=instruction_files_env.split(",") if instruction_files_env else None,
                    instructions_dir=os.getenv("SESSION_INSTRUCTIONS_DIR", ".memorymesh/instructions/"),
                    global_instruction_file=os.getenv("SESSION_GLOBAL_INSTRUCTION", ""),
                    docs_sync_enabled=os.getenv("SESSION_DOCS_SYNC_ENABLED", "true").lower() == "true",
                    docs_sync_files=docs_files_env.split(",") if docs_files_env else None,
                    max_file_size=int(os.getenv("SESSION_INSTRUCTION_MAX_FILE_SIZE", "51200")),
                ),
            ),
            instinct=InstinctConfig(
                db_path=os.getenv("INSTINCT_DB_PATH", "./db/instincts.db"),
                enabled=os.getenv("INSTINCT_ENABLED", "true").lower() == "true",
                min_samples=int(os.getenv("INSTINCT_MIN_SAMPLES", "5")),
                v2_max_instincts=int(os.getenv("INSTINCT_V2_MAX", "200")),
                max_pattern_length=int(os.getenv("INSTINCT_MAX_PATTERN_LENGTH", "200")),
                confidence_floor=float(os.getenv("INSTINCT_CONFIDENCE_FLOOR", "0.1")),
                dedup_similarity_threshold=float(os.getenv("INSTINCT_DEDUP_SIMILARITY", "0.95")),
            ),
            ecc_integration=EccIntegrationConfig(
                choke_point_enabled=os.getenv("ECC_CHOKE_POINT_ENABLED", "true").lower() == "true",
                auto_save_tool_context=os.getenv("ECC_AUTO_SAVE_TOOL_CONTEXT", "true").lower() == "true",
                bootstrap_snapshots=os.getenv("ECC_BOOTSTRAP_SNAPSHOTS", "true").lower() == "true",
            ),
            consolidation=ConsolidationConfig(
                similarity_threshold=float(os.getenv("CONSOLIDATION_SIMILARITY", "0.85")),
                min_cluster_size=int(os.getenv("CONSOLIDATION_MIN_CLUSTER", "2")),
                interval_seconds=int(os.getenv("CONSOLIDATION_INTERVAL", "3600")),
                batch_size=int(os.getenv("CONSOLIDATION_BATCH", "50")),
                enabled=os.getenv("CONSOLIDATION_ENABLED", "true").lower() == "true",
                session_memory_ttl_days=int(os.getenv("SESSION_MEMORY_TTL_DAYS", "7")),
                min_importance_to_keep=int(os.getenv("MIN_IMPORTANCE_KEEP", "3")),
                ephemeral_memory_ttl_days=int(os.getenv("EPHEMERAL_TTL_DAYS", "7")),
            ),
            level_weight_session=float(os.getenv("LEVEL_WEIGHT_SESSION", "2.0")),
            level_weight_user=float(os.getenv("LEVEL_WEIGHT_USER", "1.5")),
            level_weight_knowledge=float(os.getenv("LEVEL_WEIGHT_KNOWLEDGE", "1.0")),
            embedding=EmbeddingConfig(
            mode=os.getenv("EMBEDDING_MODE", "local"),
            model=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            remote_api_url=os.getenv("REMOTE_EMBEDDING_API_URL", ""),
            remote_api_key=os.getenv("REMOTE_EMBEDDING_API_KEY", ""),
        ),
        embedding_model=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            default_user_id=os.getenv("DEFAULT_USER_ID", "Shinn"),
            mcp_transport=os.getenv("MCP_TRANSPORT", "stdio"),
            mcp_port=int(os.getenv("MCP_PORT", "8090")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            token_budget=int(os.getenv("TOKEN_BUDGET", "1000")),
            truncation_weight_score=float(os.getenv("TRUNCATION_WEIGHT_SCORE", "0.6")),
            truncation_weight_importance=float(os.getenv("TRUNCATION_WEIGHT_IMPORTANCE", "0.3")),
            truncation_weight_recency=float(os.getenv("TRUNCATION_WEIGHT_RECENCY", "0.1")),
        )

    def validate(self):
        self.router.validate()
        self.sqlite_vec.validate()
        self.session.validate()
        self.consolidation.validate()
        self.instinct.validate()
        self.ecc_integration.validate()
