"""Deterministic synthetic 'lecture' videos for tests and CPU smoke runs.

Each lecture is a REAL mp4 (encoded with PyAV) plus a timed transcript and
three multiple-choice questions that exercise the research idea:

    q_lr      "What learning rate was used?"         on a slide; USUALLY spoken
              -> under TARGETED damage the speech is gone, so looking helps.
    q_acc     "What final accuracy is shown?"         on a slide; SOMETIMES spoken
              -> the "known visual-only fact" case when not spoken.
    q_switch  "What colour does the indicator turn?"  before/after change
              -> needs an ordered pair of frames.

``DEFAULT`` (video id ``fixture_lecture``) is a fixed lecture used by unit
tests.  ``generate_spec(seed)`` produces variations (different numbers,
colours, and whether facts are spoken) for the synthetic smoke dataset.

A registry maps video id -> spec so the test-double answerer can "read" a frame
from its true timestamp.  That double exists only to test plumbing and label
generation; synthetic results are NEVER research results.
"""

from __future__ import annotations

import dataclasses
import json
import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schemas import QAItem, Transcript, TranscriptSegment, to_jsonable

WIDTH, HEIGHT, FPS = 640, 360, 5
DURATION_S = 90.0

LR_OPTIONS = ["0.01", "0.1", "0.3", "1.0"]
ACC_OPTIONS = ["67%", "77%", "87%", "97%"]
COLOURS = {"red": (210, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 210), "yellow": (230, 190, 30)}


@dataclasses.dataclass(frozen=True)
class Slide:
    start_s: float
    end_s: float
    title: str
    body: str
    indicator: str | None = None   # colour name of the circle, for the before/after fact


@dataclasses.dataclass(frozen=True)
class LectureSpec:
    video_id: str
    slides: tuple[Slide, ...]
    speech: tuple[tuple[float, float, str], ...]
    lr: str
    acc: str
    before: str
    after: str


def _build(video_id: str, lr: str, acc: str, before: str, after: str, speak_lr: bool, speak_acc: bool,
           filler: list[str]) -> LectureSpec:
    slides = (
        Slide(0, 15, "Lecture 3", "Optimisation basics"),
        Slide(15, 30, "Setup", f"Learning rate = {lr}"),
        Slide(30, 45, "Results", f"Final accuracy: {acc}"),
        Slide(45, 55, "Circuit demo", "Switch OFF", indicator=before),
        Slide(55, 65, "Circuit demo", "Switch ON", indicator=after),
        Slide(65, 90, "Summary", "Questions?"),
    )
    speech = [
        (1.0, 6.0, "welcome to lecture three on optimisation"),
        (7.0, 13.0, filler[0]),
        (16.0, 22.0, f"for this run the learning rate was set to {lr}" if speak_lr
         else "for this run we picked the learning rate shown here"),
        (23.0, 29.0, "we kept every other setting the same"),
        (31.0, 37.0, f"here are the results the final accuracy was {acc}" if speak_acc
         else "here are the results of the experiment"),
        (38.0, 44.0, "as you can see it worked quite well"),
        (46.0, 52.0, "now a small circuit demonstration"),
        (56.0, 62.0, "watch what happens to the indicator light"),
        (66.0, 72.0, "to summarise we covered optimisation settings"),
        (74.0, 80.0, filler[1]),
        (82.0, 88.0, "see you next week"),
    ]
    return LectureSpec(video_id, slides, tuple(speech), lr, acc, before, after)


# The fixed lecture used by unit tests: lr spoken, accuracy NOT spoken, red -> green.
DEFAULT = _build("fixture_lecture", lr="0.3", acc="87%", before="red", after="green", speak_lr=True,
                 speak_acc=False, filler=["today we look at how training behaves",
                                          "please send me any questions by email"])
_REGISTRY: dict[str, LectureSpec] = {DEFAULT.video_id: DEFAULT}

# Backwards-compatible aliases for the default lecture.
SLIDES = DEFAULT.slides
SPEECH = DEFAULT.speech

_FILLER = ["today we look at how training behaves", "please send me any questions by email",
           "remember to check the reading list", "the lab session is on thursday",
           "this builds on last week's material", "we will revisit this in the exam review"]


def generate_spec(seed: int) -> LectureSpec:
    """A randomised lecture.  Probabilities are chosen so that some questions
    need vision and some do not - the controller must learn the difference."""
    rng = random.Random(seed)
    before, after = rng.sample(list(COLOURS), 2)
    spec = _build(f"synth_{seed:04d}", lr=rng.choice(LR_OPTIONS), acc=rng.choice(ACC_OPTIONS),
                  before=before, after=after, speak_lr=rng.random() < 0.85, speak_acc=rng.random() < 0.4,
                  filler=rng.sample(_FILLER, 2))
    _REGISTRY[spec.video_id] = spec
    return spec


def get_spec(video_id: str) -> LectureSpec:
    """Return the spec for a fixture video id.

    Synthetic ids encode their seed ("synth_0042"), so specs are rebuilt
    deterministically in any process.  Unknown ids raise instead of falling
    back to DEFAULT: a silent fallback once made the test double "read" the
    wrong lecture's slides in a fresh CLI process.
    """
    if video_id in _REGISTRY:
        return _REGISTRY[video_id]
    if video_id.startswith("synth_") and video_id[6:].isdigit():
        return generate_spec(int(video_id[6:]))
    raise KeyError(f"no fixture spec for video id {video_id!r}")


def slide_at(t: float, spec: LectureSpec = DEFAULT) -> Slide:
    for s in spec.slides:
        if s.start_s <= t < s.end_s:
            return s
    return spec.slides[-1]


def render_slide(slide: Slide) -> Image.Image:
    """Draw a simple slide: title bar, body text and an optional coloured indicator."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, 60], fill=(30, 60, 110))
    draw.text((20, 12), slide.title, fill=(255, 255, 255), font=ImageFont.load_default(size=34))
    draw.text((40, 150), slide.body, fill=(20, 20, 20), font=ImageFont.load_default(size=40))
    if slide.indicator:
        draw.ellipse([500, 230, 580, 310], fill=COLOURS[slide.indicator])
    return img


def write_video(path: str | Path, spec: LectureSpec = DEFAULT) -> Path:
    """Encode the lecture to ``path`` (H.264, 5 fps).  Deterministic output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[Slide, np.ndarray] = {}
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS, 1))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for i in range(int(DURATION_S * FPS)):
            slide = slide_at(i / FPS, spec)
            if slide not in cache:
                cache[slide] = np.asarray(render_slide(slide))
            for packet in stream.encode(av.VideoFrame.from_ndarray(cache[slide], format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)
    return path


def transcript(video_id: str = DEFAULT.video_id) -> Transcript:
    spec = get_spec(video_id)
    segs = [TranscriptSegment(id=f"s{i:05d}", start_s=a, end_s=b, text=t) for i, (a, b, t) in enumerate(spec.speech)]
    return Transcript(video_id=spec.video_id, segments=segs, source="fixture")


def qa_items(video_id: str = DEFAULT.video_id) -> list[QAItem]:
    spec = get_spec(video_id)
    colours = list(COLOURS)
    common = dict(video_id=spec.video_id, source_dataset="synthetic_lecture", source_split="unassigned",
                  provenance_and_license="synthetic, generated by videoqa.fixtures (no third-party content)")
    return [
        QAItem(qa_id=f"{spec.video_id}/q_lr", question="What learning rate was used for this run?",
               gold_answer=spec.lr, options=LR_OPTIONS, gold_option_index=LR_OPTIONS.index(spec.lr),
               evidence_intervals_s=[(15.0, 30.0)], **common),
        QAItem(qa_id=f"{spec.video_id}/q_acc", question="What final accuracy is shown on the results slide?",
               gold_answer=spec.acc, options=ACC_OPTIONS, gold_option_index=ACC_OPTIONS.index(spec.acc),
               evidence_intervals_s=[(30.0, 45.0)], **common),
        QAItem(qa_id=f"{spec.video_id}/q_switch",
               question="What colour does the indicator light turn to in the circuit demo?",
               gold_answer=spec.after, options=colours, gold_option_index=colours.index(spec.after),
               evidence_intervals_s=[(45.0, 65.0)], **common),
    ]


def write_dataset(out_dir: str | Path, n_videos: int, seed0: int = 0) -> Path:
    """Write ``n_videos`` synthetic lectures: videos/*.mp4, transcripts/*.json and
    qa.jsonl.  Returns the path of qa.jsonl."""
    out = Path(out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    lines = []
    for k in range(n_videos):
        spec = generate_spec(seed0 + k)
        write_video(out / "videos" / f"{spec.video_id}.mp4", spec)
        segs = [{"start": a, "end": b, "text": t} for a, b, t in spec.speech]
        (out / "transcripts" / f"{spec.video_id}.json").write_text(json.dumps(segs), encoding="utf-8")
        lines += [json.dumps(to_jsonable(qa)) for qa in qa_items(spec.video_id)]
    qa_path = out / "qa.jsonl"
    qa_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return qa_path
