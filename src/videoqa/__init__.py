"""videoqa: low-cost, evidence-grounded long-video question answering.

Pipeline at a glance (each stage lives in its own module):

    transcript.py  -> timed speech segments + reliability features
    damage.py      -> CLEAN / TARGETED_DAMAGE / MATCHED_CONTROL_DAMAGE transcript variants
    retrieval.py   -> cheap candidate time windows (BM25 + optional dense)
    frames.py      -> decode real frames at requested times (true PTS)
    scout.py       -> frozen visual scout emitting ONLY non-answer-bearing signals
    controller.py  -> small "brain" choosing LOOK / LOOK_ELSEWHERE / EXPAND / STOP
    packing.py     -> concise transcript excerpt that keeps numbers/negations
    answerer.py    -> frozen final VLM that sees real frames + excerpt
    costs.py       -> metering of every expensive action
    pipeline.py    -> glues the stages together and returns answer + cost trace
"""

__version__ = "0.1.0"
