"""
FactLens — db/cli.py

Command-line tool for schema management.

Usage
-----
    python -m db.cli               # deploy schema (idempotent)
    python -m db.cli --reset       # DROP all tables then redeploy (destructive!)
"""

from __future__ import annotations

import argparse
import logging
import sys

from db.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="FactLens DB schema tool")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables then redeploy (DESTRUCTIVE — dev only)",
    )
    args = parser.parse_args()

    with Database() as db:
        if args.reset:
            confirm = input("This will DELETE all data. Type 'yes' to continue: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                sys.exit(0)
            db.reset_schema()
        else:
            db.deploy_schema()


if __name__ == "__main__":
    main()
