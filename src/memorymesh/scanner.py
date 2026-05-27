"""CodebaseScanner — auto-scan workspace for cross-session context."""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                ".pytest_cache", ".vscode", "memory-mesh-env", "__pycache__"}
EXCLUDE_FILES = {".pyc", ".pyo", ".so", ".dll", ".exe", ".gitkeep"}
TEXT_EXTENSIONS = {".py", ".toml", ".json", ".jsonc", ".md", ".txt",
                   ".yml", ".yaml", ".cfg", ".ini", ".env.example",
                   "Dockerfile", "Makefile", ".gitignore"}

SCAN_MAX_DEPTH = 3
SCAN_MAX_FILES_PER_DIR = 20
SCAN_MAX_BYTES = 100_000


class CodebaseScanner:
    def __init__(self, workspace_path: str = ""):
        self.workspace_path = workspace_path or os.getcwd()

    def _should_exclude_dir(self, name: str) -> bool:
        return name in EXCLUDE_DIRS or name.startswith(".")

    def _is_text_file(self, path: str) -> bool:
        _, ext = os.path.splitext(path)
        return ext in TEXT_EXTENSIONS

    def _scan_tree(self, path: str, depth: int = 0) -> List[Dict[str, Any]]:
        if depth > SCAN_MAX_DEPTH:
            return []
        entries = []
        try:
            names = sorted(os.listdir(path))
        except PermissionError:
            return entries
        count = 0
        for name in names:
            if count >= SCAN_MAX_FILES_PER_DIR:
                break
            full = os.path.join(path, name)
            if os.path.isdir(full):
                if self._should_exclude_dir(name):
                    continue
                children = self._scan_tree(full, depth + 1)
                entries.append({"type": "dir", "name": name, "children": children})
                count += 1
            else:
                if any(name.endswith(s) for s in EXCLUDE_FILES):
                    continue
                size = os.path.getsize(full) if os.path.exists(full) else 0
                entry: Dict[str, Any] = {"type": "file", "name": name, "size": size}
                if size < SCAN_MAX_BYTES and self._is_text_file(full):
                    try:
                        with open(full, "r", encoding="utf-8", errors="replace") as f:
                            entry["preview"] = f.read(2000)
                    except Exception:
                        pass
                entries.append(entry)
                count += 1
        return entries

    def _read_key_files(self) -> Dict[str, str]:
        from .utils.path_sanitizer import safe_join_and_validate
        results = {}
        key_files = [
            "README.md", "Makefile", ".env.example",
            "pyproject.toml", ".opencode.json",
        ]
        for fname in key_files:
            try:
                path = safe_join_and_validate(self.workspace_path, fname)
                if os.path.exists(path) and os.path.getsize(path) < SCAN_MAX_BYTES:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        results[fname] = f.read(2000)
            except Exception:
                pass
        return results

    def scan(self) -> Dict[str, Any]:
        tree = self._scan_tree(self.workspace_path)
        key_files = self._read_key_files()
        summary_parts = [f"Workspace: {self.workspace_path}"]
        summary_parts.append(f"Total top-level entries: {len(tree)}")
        for entry in tree:
            summary_parts.append(f"  {entry['type']}: {entry['name']}")
        if key_files:
            summary_parts.append("\nKey files found:")
            for name in key_files:
                summary_parts.append(f"  {name}")
        return {
            "workspace_path": self.workspace_path,
            "tree": tree,
            "key_files": key_files,
            "summary": "\n".join(summary_parts),
        }
