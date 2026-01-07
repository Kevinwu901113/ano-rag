import os
import subprocess
import pytest
from pathlib import Path

def test_no_legacy_imports():
    """
    Gatekeeper test: Scan relrag/ directory for forbidden top-level imports.
    Use relrag.* imports instead of legacy top-level packages.
    """
    relrag_dir = Path(__file__).resolve().parent.parent
    
    # We use grep to find forbidden patterns.
    # Block legacy top-level package imports (schema/utils/retriever/etc).
    # The test matches explicit "from <pkg>." or "import <pkg>" statements.
    
    forbidden_pkgs = [
        "schema",
        "utils",
        "retriever",
        "telemetry",
        "config",
        "generator",
        "indexer",
        "pipeline",
        "postprocess",
        "doc",
        "validators",
        "rag_core",
        "structrag",
    ]
    
    for pkg in forbidden_pkgs:
        # Check "from pkg."
        # We search recursively in relrag/
        # We use python to walk and check file content to avoid shell grep nuances
        
        for root, _, files in os.walk(relrag_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, file)
                if "tests" in file_path: # Skip tests if needed, but better to check all code
                    pass
                
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Check for "from <pkg>." or "import <pkg>".
                    # Allow "from relrag.<pkg>..." forms.
                    
                    # Pattern 1: from pkg.
                    if stripped.startswith(f"from {pkg}."):
                        pytest.fail(f"Forbidden import in {file_path}:{i+1}: {stripped}")
                    
                    # Pattern 2: import pkg
                    if stripped.startswith(f"import {pkg}"):
                        # Be careful with similar module names (e.g., schematics vs schema).
                        parts = stripped.split()
                        if len(parts) >= 2 and parts[1] == pkg:
                             pytest.fail(f"Forbidden import in {file_path}:{i+1}: {stripped}")
                        elif len(parts) >= 2 and parts[1].startswith(f"{pkg}."):
                             pytest.fail(f"Forbidden import in {file_path}:{i+1}: {stripped}")

    # Explicit check for the user mentioned file
    note_validator = relrag_dir / "validators/note_validator.py"
    if note_validator.exists():
        with open(note_validator, "r") as f:
            content = f.read()
        forbidden = "from " + "schema." + "vocabulary"
        if forbidden in content:
            pytest.fail("relrag/validators/note_validator.py still contains legacy schema.vocabulary import")

if __name__ == "__main__":
    test_no_legacy_imports()
