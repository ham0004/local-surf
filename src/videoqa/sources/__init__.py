"""Per-dataset adapters that convert an official source into the local layout
consumed by :mod:`videoqa.datasets` (``qa.jsonl`` + ``videos/`` + ``transcripts/``).

Each adapter is read-only with respect to the official source: it never
invents fields, and it records how confident its evidence extraction is
(e.g. whether a quoted subtitle span was found) so downstream code can filter
on that confidence instead of silently trusting a heuristic.
"""
