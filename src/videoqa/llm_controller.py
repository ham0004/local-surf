"""Small text-LLM controller: Qwen3-0.6B + LoRA with a scalar gain head.

Input: serialize.serialize(obs, action) - allowlisted observables only.
Output: predicted answer-quality gain of that action (one scalar per text,
read from the last non-pad token's hidden state through a linear head).

The decision rule is the one every utility controller uses:
    net = predicted_gain - lambda * cost_ms / 1000, STOP if max net <= 0,
with lambda calibrated on dev (train_assay.simulate, same labels as the MLP).

Training losses are exactly the MLP's V0 / V1 / V2 (train_assay.py), applied
per mini-batch of exact-key triplets plus random extra rows (EXPAND rows only
ever enter the absolute term).

Frozen for the main experiment: the scout and the final answerer. Only the
LoRA adapters and the head are trained here.

Model: Qwen/Qwen3-0.6B, pinned revision below (Apache-2.0 per the model card;
verify before redistribution).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from .assay import AssayRow, read_assay
from .cost_model import DEFAULT, CostModel
from .features import LegalAction
from .schemas import Action, ActionKind, ControllerObservation
from .serialize import serialize
from .train_assay import LAMBDA_GRID, simulate, triplet_terms

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclasses.dataclass
class LLMTrainConfig:
    variant: str = "V1"
    seed: int = 0
    steps: int = 300
    batch_triplets: int = 8
    batch_extra: int = 8
    lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    max_len: int = 384
    huber_delta: float = 0.5
    w_pair: float = 1.0
    w_clean: float = 0.5
    w_rank: float = 0.2


def _torch():
    import torch  # noqa: PLC0415
    return torch


class LLMScorer:
    """Base LM (frozen) + LoRA + linear head. Built lazily so importing this
    module needs no torch."""

    def __init__(self, cfg: LLMTrainConfig, device: str = "cuda", adapter_dir: str | None = None) -> None:
        torch = _torch()
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        torch.manual_seed(cfg.seed)
        self.cfg, self.device = cfg, device
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.tok.padding_side = "right"
        base = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16).to(device)
        if adapter_dir:
            self.model = PeftModel.from_pretrained(base, adapter_dir)
        else:
            lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"])
            # Activation memory, not weights, dominates at batch 32 x ~300 tokens:
            # recompute activations in backward (measured 13.5 GB peak without).
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
            self.model = get_peft_model(base, lcfg)
        hidden = base.config.hidden_size
        self.head = torch.nn.Linear(hidden, 1).to(device=device, dtype=torch.float32)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        if adapter_dir and (Path(adapter_dir) / "head.pt").exists():
            self.head.load_state_dict(torch.load(Path(adapter_dir) / "head.pt", map_location=device))

    def trainable_params(self) -> int:
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return n + sum(p.numel() for p in self.head.parameters())

    def forward(self, texts: list[str]):
        torch = _torch()
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.cfg.max_len).to(self.device)
        out = self.model(**enc).last_hidden_state                        # (B, T, H)
        last = enc["attention_mask"].sum(dim=1) - 1
        h = out[torch.arange(out.shape[0], device=self.device), last].float()
        return self.head(h).squeeze(-1), int(enc["attention_mask"].sum())

    def predict(self, texts: list[str], batch: int = 32) -> np.ndarray:
        torch = _torch()
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                u, _ = self.forward(texts[i:i + batch])
                outs.append(u.cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0)

    def save(self, out_dir: str | Path, meta: dict) -> None:
        torch = _torch()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        torch.save(self.head.state_dict(), out / "head.pt")
        (out / "controller_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _huber(r, delta):
    torch = _torch()
    a = r.abs()
    return torch.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))


def train_llm(train_dir: str | Path, dev_dir: str | Path, out_dir: str | Path, cfg: LLMTrainConfig,
              price_per_s: float, device: str = "cuda", log=print) -> dict:
    torch = _torch()
    tr_rows, tr_trip = read_assay(train_dir)
    dv_rows, _ = read_assay(dev_dir)
    if any(r.experiment_split == "test" for r in tr_rows + dv_rows):
        raise ValueError("test rows must never be used for training or calibration")
    if {r.video_id for r in tr_rows} & {r.video_id for r in dv_rows}:
        raise ValueError("train/dev share videos")
    if not all(r.obs_text for r in tr_rows + dv_rows):
        raise ValueError("assay rows lack obs_text (labels built before serialize.py); rebuild them")

    rng = np.random.default_rng(cfg.seed)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    scorer = LLMScorer(cfg, device)
    params = [p for p in scorer.model.parameters() if p.requires_grad] + list(scorer.head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    texts = [r.obs_text for r in tr_rows]
    y_all = np.asarray([r.gain for r in tr_rows], dtype=np.float32)
    tc, tk, ck, _ = triplet_terms(tr_rows, tr_trip)
    trip = np.stack([tc[:, 0], tc[:, 1], tk[:, 1]], axis=1) if len(tc) else np.zeros((0, 3), int)  # T, C, K
    curve, tokens_seen, t0 = [], 0, time.perf_counter()
    scorer.model.train()
    for step in range(cfg.steps):
        pick = rng.choice(len(trip), size=min(cfg.batch_triplets, len(trip)), replace=False) if len(trip) else []
        tri = trip[pick] if len(pick) else np.zeros((0, 3), int)
        extra = rng.choice(len(tr_rows), size=min(cfg.batch_extra, len(tr_rows)), replace=False)
        idx = np.concatenate([tri.reshape(-1), extra]).astype(int)
        u, ntok = scorer.forward([texts[i] for i in idx])
        tokens_seen += ntok
        y = torch.tensor(y_all[idx], device=device)
        loss = _huber(u - y, cfg.huber_delta).mean()
        n = len(tri)
        if n and cfg.variant in ("V1", "V2"):
            ut, uc, uk = u[0:3 * n:3], u[1:3 * n:3], u[2:3 * n:3]
            yt, yc, yk = y[0:3 * n:3], y[1:3 * n:3], y[2:3 * n:3]
            loss = loss + cfg.w_pair * _huber((ut - uc) - (yt - yc), cfg.huber_delta).mean()
            if cfg.variant == "V2":
                loss = loss + cfg.w_clean * (_huber((ut - uk) - (yt - yk), cfg.huber_delta).mean()
                                             + _huber((uc - uk) - (yc - yk), cfg.huber_delta).mean())
                differ = yt != yc
                if differ.any():
                    sign = torch.sign(yt - yc)[differ]
                    d = (ut - uc)[differ] * sign
                    loss = loss + cfg.w_rank * torch.nn.functional.softplus(-d).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
        if log and (step % 50 == 0 or step == cfg.steps - 1):
            log(f"{cfg.variant} seed{cfg.seed} step {step}: loss {np.mean(curve[-50:]):.4f}")
    train_s = time.perf_counter() - t0

    # Dev: predictions, calibration, metrics (same definitions as the MLP)
    t1 = time.perf_counter()
    pred = scorer.predict([r.obs_text for r in dv_rows])
    infer_ms_per_action = 1000 * (time.perf_counter() - t1) / max(1, len(dv_rows))
    _, dv_trip = read_assay(dev_dir)
    curve_l = [{"lambda": lam, **simulate(pred, dv_rows, lam, price_per_s)["all"]} for lam in LAMBDA_GRID]
    best = max(range(len(curve_l)), key=lambda i: (round(curve_l[i]["utility"], 12), curve_l[i]["lambda"]))
    lam = curve_l[best]["lambda"]
    yd = np.asarray([r.gain for r in dv_rows])
    metrics = {"gain_mae": float(np.abs(pred - yd).mean())}
    if dv_trip:
        t = np.array([x["targeted"] for x in dv_trip])
        c = np.array([x["control"] for x in dv_trip])
        metrics["tc_diff_mae"] = float(np.abs((pred[t] - pred[c]) - (yd[t] - yd[c])).mean())
        differ = yd[t] != yd[c]
        metrics["tc_rank_n"] = int(differ.sum())
        metrics["tc_rank_acc"] = (float((np.sign(pred[t] - pred[c])[differ] == np.sign(yd[t] - yd[c])[differ]).mean())
                                  if differ.any() else None)
    meta = {"model_id": MODEL_ID, "revision": REVISION, "config": dataclasses.asdict(cfg),
            "lambda_per_s": lam, "price_per_s": price_per_s,
            "trainable_params": scorer.trainable_params(), "train_tokens": tokens_seen,
            "train_seconds": round(train_s, 1), "final_loss": float(np.mean(curve[-20:])),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
            "dev_infer_ms_per_action_batched": round(infer_ms_per_action, 2),
            "dev": metrics, "dev_at_lambda": simulate(pred, dv_rows, lam, price_per_s), "pareto": curve_l}
    scorer.save(Path(out_dir) / f"llm_{cfg.variant}_seed{cfg.seed}", meta)
    del scorer
    if device == "cuda":
        torch.cuda.empty_cache()
    return meta


class LLMController:
    """Utility controller whose gain estimate comes from the LLM scorer."""

    name = "llm"
    scout_seed: str | int = 4

    def __init__(self, adapter_dir: str | Path, cost_model: CostModel = DEFAULT, device: str = "cuda") -> None:
        meta = json.loads((Path(adapter_dir) / "controller_meta.json").read_text(encoding="utf-8"))
        cfg = LLMTrainConfig(**meta["config"])
        self.scorer = LLMScorer(cfg, device, adapter_dir=str(adapter_dir))
        self.cost_model = cost_model.with_lambda(meta["lambda_per_s"])
        self.name = f"llm[{Path(adapter_dir).name}]"

    def decide(self, obs: ControllerObservation, legal: list[LegalAction]) -> Action:
        acts = [a for a in legal if a.kind != ActionKind.STOP]
        if not acts:
            return Action(ActionKind.STOP, reason="NO_LEGAL_ACTION")
        by_id = {c.id: c for c in obs.candidates}
        texts = [serialize(obs, a.kind, by_id.get(a.candidate_id)) for a in acts]
        gains = self.scorer.predict(texts)
        net = [self.cost_model.net(float(g), a.cost_ms) for g, a in zip(gains, acts, strict=True)]
        b = int(np.argmax(net))
        if not math.isfinite(net[b]) or net[b] <= 0.0:
            return Action(ActionKind.STOP, predicted_utility=float(gains[b]), reason="NO_ACTION_WORTH_COST")
        a = acts[b]
        return Action(a.kind, a.candidate_id, predicted_utility=float(gains[b]), reason="LLM_GAIN")
