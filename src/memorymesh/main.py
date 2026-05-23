"""Entry point for MemoryMesh MCP server."""
import os
import sys
import subprocess
from pathlib import Path


OPencode_JSON_CONTENT = """\
{{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {{
    "memorymesh": {{
      "type": "local",
      "command": [
        "{python}",
        "-m",
        "memorymesh"
      ],
      "environment": {{
        "ROUTER_URL": "http://127.0.0.1:20128/v1",
        "DEFAULT_MODEL": "ds/deepseek-v4-flash",
        "FALLBACK_MODEL": "ds/deepseek-v4-pro",
        "EMBEDDING_MODEL": "paraphrase-multilingual-MiniLM-L12-v2",
        "VEC_DB_PATH": "{vec_db}",
        "VEC_AUTO_MIGRATE": "false",
        "SESSION_DB_PATH": "{session_db}",
        "DEFAULT_USER_ID": "{user_id}",
        "MCP_TRANSPORT": "stdio",
        "LOG_LEVEL": "INFO"
      }},
      "enabled": true
    }}
  }}
}}
"""


def cmd_init(args):
    """Initialize a new MemoryMesh workspace with OpenCode integration."""
    target = Path(args[0]).resolve() if args else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    # ── 1. Install package if not done ──
    try:
        import memorymesh  # noqa: F401
    except ImportError:
        print("[ .. ] Package not installed — running pip install -e . ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(target)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"[fail] pip install failed:\n{result.stderr}")
            raise SystemExit(1)
        print("[ ok ] Package installed")
        # Reload so the import works after install
        import importlib
        importlib.invalidate_caches()

    # ── 2. db/ directory ──
    db_dir = target / "db"
    db_dir.mkdir(exist_ok=True)

    # ── 3. .env file ──
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

    # ── 4. .opencode/ directory ──
    opencode_dir = target / ".opencode"
    opencode_dir.mkdir(exist_ok=True)

    # ── 5. .opencode/opencode.json (dynamic, with sys.executable) ──
    python_path = sys.executable.replace("\\", "/")
    user_id = os.environ.get("DEFAULT_USER_ID") or target.name
    opencode_json_path = opencode_dir / "opencode.json"

    json_content = OPencode_JSON_CONTENT.format(
        python=python_path,
        vec_db=(db_dir / "memory.db").as_posix(),
        session_db=(db_dir / "sessions.db").as_posix(),
        user_id=user_id,
    )
    opencode_json_path.write_text(json_content)
    print(f"[ ok ] MCP config created at {opencode_json_path}")

    print()
    print("  ── Setup complete ──")
    print()
    print(f"  Database:  ./db/")
    print(f"  Config:    {env_file}")
    print()
    print("  [OpenCode Users]")
    print("  Config auto-created at .opencode/opencode.json. Just run: opencode")
    print()
    print("  [Cursor / Cline / Continue Users]")
    print("  Add this MCP Server to your IDE settings:")
    print(f"  Command: {python_path} -m memorymesh")


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
