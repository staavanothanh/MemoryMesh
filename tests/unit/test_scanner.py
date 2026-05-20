import pytest
import os
import tempfile
from memorymesh.scanner import CodebaseScanner


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "src", "app"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "docs"), exist_ok=True)
    os.makedirs(os.path.join(tmp, ".git"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "__pycache__"), exist_ok=True)

    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write("# Test Project\nA test project for scanning.")
    with open(os.path.join(tmp, "pyproject.toml"), "w") as f:
        f.write("[project]\nname = \"test\"\ndependencies = [\"pytest\"]\n")
    with open(os.path.join(tmp, "src", "app", "main.py"), "w") as f:
        f.write("def main():\n    pass\n")
    with open(os.path.join(tmp, "src", "app", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(tmp, "docs", "guide.md"), "w") as f:
        f.write("# Guide\nHow to use.")
    yield tmp
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_scanner_scans_structure(temp_workspace):
    scanner = CodebaseScanner(workspace_path=temp_workspace)
    result = scanner.scan()

    assert result["workspace_path"] == temp_workspace
    assert "tree" in result
    assert "key_files" in result
    assert "summary" in result


@pytest.mark.asyncio
async def test_scanner_excludes_git(temp_workspace):
    scanner = CodebaseScanner(workspace_path=temp_workspace)
    result = scanner.scan()

    def find_name(entries, name):
        for e in entries:
            if e["name"] == name:
                return True
            if "children" in e:
                if find_name(e["children"], name):
                    return True
        return False

    assert not find_name(result["tree"], ".git")
    assert not find_name(result["tree"], "__pycache__")


@pytest.mark.asyncio
async def test_scanner_reads_key_files(temp_workspace):
    scanner = CodebaseScanner(workspace_path=temp_workspace)
    result = scanner.scan()

    assert "README.md" in result["key_files"]
    assert "pyproject.toml" in result["key_files"]
    assert "Test Project" in result["key_files"]["README.md"]


@pytest.mark.asyncio
async def test_scanner_summary_contains_info(temp_workspace):
    scanner = CodebaseScanner(workspace_path=temp_workspace)
    result = scanner.scan()

    assert temp_workspace in result["summary"]
    assert "README.md" in result["summary"]


@pytest.mark.asyncio
async def test_scanner_default_path(temp_workspace):
    original = os.getcwd()
    try:
        os.chdir(temp_workspace)
        scanner = CodebaseScanner()
        result = scanner.scan()
        assert result["workspace_path"] == temp_workspace
    finally:
        os.chdir(original)


@pytest.mark.asyncio
async def test_scanner_empty_directory():
    tmp = tempfile.mkdtemp()
    try:
        scanner = CodebaseScanner(workspace_path=tmp)
        result = scanner.scan()
        assert len(result["tree"]) == 0
        assert len(result["key_files"]) == 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_scanner_shows_files_and_dirs(temp_workspace):
    scanner = CodebaseScanner(workspace_path=temp_workspace)
    result = scanner.scan()

    dirs = [e for e in result["tree"] if e["type"] == "dir"]
    files = [e for e in result["tree"] if e["type"] == "file"]

    assert any(d["name"] == "src" for d in dirs)
    assert any(f["name"] == "README.md" for f in files)
    assert any(f["name"] == "pyproject.toml" for f in files)
