"""Post the sample tickets to a running service.

    python scripts/load_samples.py [samples/tickets.json] [--url http://localhost:8000]

Safe to run repeatedly: re-posting an existing ticket is an idempotent no-op (200).
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="samples/tickets.json")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        tickets = json.load(f)

    for ticket in tickets:
        request = urllib.request.Request(
            f"{args.url}/tickets",
            data=json.dumps(ticket).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        print(f"{ticket['id']}: {status}")


if __name__ == "__main__":
    main()
