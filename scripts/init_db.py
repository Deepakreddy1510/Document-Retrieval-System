from __future__ import annotations

import sys
from pathlib import Path

ROOT = ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pdf_rag.config import settings  # noqa: E402
from pdf_rag.database import Database  # noqa: E402


def main() -> None:
    db = Database(settings.database_url)
    db.initialize_schema()
    print("Database schema initialized successfully.")
    print(f"Connection healthy: {db.healthcheck()}")


if __name__ == "__main__":
    main()
