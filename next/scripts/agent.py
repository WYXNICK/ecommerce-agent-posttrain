#!/usr/bin/env python3
"""Run the isolated implementation from any working directory."""
from pathlib import Path
import os
import sys

os.environ.setdefault("HF_HOME", "/tmp/shopping-next-cache/huggingface")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopping_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
