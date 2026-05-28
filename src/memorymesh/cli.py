"""CLI tools for MemoryMesh — stateless DB queries with lazy UI imports.

All commands follow the Fetch-and-Release pattern:
1. Open DB → query all data → close DB
2. Then render UI (with lazy-imported rich)
"""

import argparse
import json
import os
import sys


def _get_db_paths():
    from .config import AppConfig
    config = AppConfig.from_env()
    return config.session.db_path, config.sqlite_vec.db_path, config.instinct.db_path


def _open_db(db_path: str):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_sessions(args):
    """List all sessions — fetch-and-release pattern."""
    db_path, _, _ = _get_db_paths()
    import sqlite3
    try:
        conn = _open_db(db_path)
        cursor = conn.execute(
            """SELECT session_id, user_id, status, workspace_path, created_at, updated_at, ended_at
               FROM sessions ORDER BY created_at DESC LIMIT ?""",
            (args.limit,),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
    except (sqlite3.OperationalError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Lazy import rich for UI rendering
    from rich.console import Console
    from rich.table import Table

    console = Console()
    if not rows:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title=f"Sessions (last {len(rows)})")
    table.add_column("ID", style="cyan")
    table.add_column("User", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Workspace")
    table.add_column("Created")
    table.add_column("Updated")

    for r in rows:
        sid = r["session_id"][:8] + "..." if r["session_id"] else "?"
        status_style = "green" if r["status"] == "active" else "white"
        table.add_row(
            sid,
            r.get("user_id", "?")[:12],
            f"[{status_style}]{r['status']}[/{status_style}]",
            r.get("workspace_path", "")[:20],
            (r.get("created_at", "") or "")[:19],
            (r.get("updated_at", "") or "")[:19],
        )
    console.print(table)


def cmd_stats(args):
    """Show system statistics — fetch-and-release pattern."""
    session_db, vec_db, instinct_db = _get_db_paths()
    import sqlite3
    stats = {}

    # Session DB stats
    try:
        conn = _open_db(session_db)
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM sessions")
        stats["sessions"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM sessions WHERE status='active'")
        stats["active_sessions"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM context_log")
        stats["context_log_entries"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM raw_log")
        stats["raw_log_entries"] = cursor.fetchone()["cnt"]
        conn.close()
    except (sqlite3.OperationalError, FileNotFoundError):
        stats["sessions"] = stats["active_sessions"] = stats["context_log_entries"] = stats["raw_log_entries"] = 0

    # Memory DB stats
    try:
        conn = _open_db(vec_db)
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE deleted=0")
        stats["memories"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM entities")
        stats["entities"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM relations")
        stats["relations"] = cursor.fetchone()["cnt"]
        conn.close()
    except (sqlite3.OperationalError, FileNotFoundError):
        stats["memories"] = stats["entities"] = stats["relations"] = 0

    # Instinct DB stats
    try:
        conn = _open_db(instinct_db)
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM instincts WHERE active=1")
        stats["instincts"] = cursor.fetchone()["cnt"]
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM instincts_v2")
        stats["instincts_v2"] = cursor.fetchone()["cnt"]
        conn.close()
    except (sqlite3.OperationalError, FileNotFoundError):
        stats["instincts"] = stats["instincts_v2"] = 0

    # Lazy import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    table = Table(title="MemoryMesh Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Sessions (total)", str(stats.get("sessions", 0)))
    table.add_row("  Active", str(stats.get("active_sessions", 0)))
    table.add_row("Context log entries", str(stats.get("context_log_entries", 0)))
    table.add_row("Raw log entries", str(stats.get("raw_log_entries", 0)))
    table.add_row("Memories", str(stats.get("memories", 0)))
    table.add_row("Graph entities", str(stats.get("entities", 0)))
    table.add_row("Graph relations", str(stats.get("relations", 0)))
    table.add_row("Instincts (v1)", str(stats.get("instincts", 0)))
    table.add_row("Instincts (v2)", str(stats.get("instincts_v2", 0)))

    # DB sizes
    for label, path in [("Session DB", session_db), ("Memory DB", vec_db), ("Instinct DB", instinct_db)]:
        try:
            size = os.path.getsize(path)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            table.add_row(f"  {label} size", size_str)
        except OSError:
            pass

    console.print(Panel(table, title="[bold]MemoryMesh[/bold]", border_style="blue"))


def main():
    parser = argparse.ArgumentParser(prog="memorymesh", description="MemoryMesh MCP Server & CLI")
    parser.add_argument("--version", action="version", version="memorymesh 0.3.0")

    sub = parser.add_subparsers(dest="command")

    # memorymesh sessions
    sessions_parser = sub.add_parser("sessions", help="List sessions")
    sessions_parser.add_argument("--limit", type=int, default=20, help="Max sessions to show (default: 20)")

    # memorymesh stats
    sub.add_parser("stats", help="Show system statistics")

    # memorymesh init (existing)
    init_parser = sub.add_parser("init", help="Initialize a workspace")
    init_parser.add_argument("path", nargs="?", default=None, help="Target directory")

    # memorymesh serve
    serve_parser = sub.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    serve_parser.add_argument("--port", type=int, default=8090, help="Port")

    args = parser.parse_args()

    if args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "init":
        from .main import cmd_init
        cmd_init([args.path] if args.path else [])
    elif args.command == "serve":
        from .main import main as server_main
        server_main()
    else:
        parser.print_help()
