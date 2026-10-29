"""
CLI entrypoint forwarding to app.services.ingest.
Run with:
    python ingest.py
    python ingest.py --file documents/handbook.pdf
    python ingest.py --url https://synkcorp.com/pricing
"""

import runpy

if __name__ == "__main__":
    runpy.run_module("app.services.ingest", run_name="__main__")
