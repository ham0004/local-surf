"""Framework v3: open-ended answers, learned temporal proposals (Head A), set selection (Head B).

See docs/v3/research_plan.md. Modules:

    openended.py   open-ended answer prompt, answerer and request-identity answer cache
    judge.py       local LLM judge (consistency + coverage vs a reference) with a cache
"""
