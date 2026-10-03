"""Framework 2: dual-path candidate proposal + learned, context-dependent frame selection.

Stages (one module each, composed in ``pilot.py``):

    candidates.py   candidate windows (BM25) -> Path A (hot moments) + Path B (visual scan)
                    -> merged, de-duplicated frame pool
    encoders.py     frozen, cached features per frame: MobileCLIP image embedding + OCR text
    head_a.py       transcript hot-moment scorer with TWO outputs:
                    usefulness of adding a transcript segment / of adding its nearby frame
    teacher.py      frozen Qwen3-VL answerer as the label source, with an exact-identity cache
    labeling.py     utility labels: independent (single-frame) vs prefix-chain (history-dependent)
    head_b.py       small frame scorer: utility(candidate | question, transcript, selected history)
    selectors.py    selection strategies behind one interface (similarity, MMR, unary, greedy)
    pilot.py        the controlled pilot: audit, label, train, evaluate, under a GPU-time budget

Everything heavy (answerer, MobileCLIP, OCR) is frozen; only the two small heads train.
"""
