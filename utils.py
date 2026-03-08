"""
utils.py
Helper utilities for the Smart Contract Assistant.
"""

import os


def format_sources(docs: list) -> list:
    """
    Extract unique source references from retrieved documents.
    Includes filename and page number where available.
    """
    seen = set()
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "contract")
        page   = doc.metadata.get("page")
        filename = os.path.basename(source) if source else "contract"

        label = f"{filename} (Page {int(page) + 1})" if page is not None else filename

        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources if sources else ["contract"]
