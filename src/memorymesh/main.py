"""Entry point for MemoryMesh MCP server."""
import os
import sys
from pathlib import Path


def cmd_init(args):
    """Initialize a new MemoryMesh workspace."""
    target = Path(args[0]).resolve() if args else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    db_dir = target / "db"
    db_dir.mkdir(exist_ok=True)

    env_file = target / ".env"
    if env_file.exists():
        print(f"[skip] .env already exists at {env_file}")
    else:
        folder_name = target.name
        env_content = f"""# MemoryMesh configuration - created by `memorymesh init`
# Adjust these values for your environment.

ROUTER_URL=http://127.0.0.1:20128/v1
DEFAULT_MODEL=your-model
FALLBACK_MODEL=your-fallback-model
ROUTER_TIMEOUT=30
ROUTER_RETRIES=3

EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

VEC_DB_PATH={db_dir / "memory.db"}
SESSION_DB_PATH={db_dir / "sessions.db"}
INSTINCT_DB_PATH={db_dir / "instincts.db"}
VEC_AUTO_MIGRATE=false

DEFAULT_USER_ID={folder_name}
MAX_MEMORY_LENGTH=2000

MCP_TRANSPORT=stdio
MCP_PORT=8090

LOG_LEVEL=INFO
"""
        env_file.write_text(env_content)
        print(f"[ ok ] .env created at {env_file}")

    print(f"[ ok ] db directory at {db_dir}")
    print()
    print("  Next steps:")
    print(f"    1. Edit {env_file} to set your LLM endpoint and model")
    print(f"    2. Run:  python -m memorymesh")
    print(f"    3. Or:   python -m memorymesh --mcp-transport sse")


def main():
    argv = sys.argv[1:] if hasattr(sys, "argv") else []

    if argv and argv[0] == "init":
        cmd_init(argv[1:])
        return

    # Default: start the MCP server
    from memorymesh.mcp_server.server import main as server_main
    server_main()


if __name__ == "__main__":
    main()
