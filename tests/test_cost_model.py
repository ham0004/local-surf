"""One cost definition: cached work is free, scouting costs extra, net = gain - lambda*cost."""

from videoqa.cost_model import DEFAULT, load_cost_model


def test_look_cost_counts_only_uncached_work():
    cold = DEFAULT.look_ms(visual_tokens=220, decoded=False, scouted=False, will_scout=True)
    warm = DEFAULT.look_ms(visual_tokens=220, decoded=True, scouted=True, will_scout=True)
    assert cold == 40 + 45 + 18 + 0.36 * 220
    assert warm == 0.36 * 220          # only the answerer's extra prefill remains


def test_net_uses_lambda_per_second():
    assert DEFAULT.with_lambda(0.5).net(gain=0.2, cost_ms=200) == 0.2 - 0.1
    assert DEFAULT.net(0.0, 0.0) == 0.0


def test_measured_file_loads_and_is_marked_measured():
    cm = load_cost_model("configs/cost_model_rtx5060ti.yaml", lambda_per_s=0.3)
    assert cm.measured and cm.lambda_per_s == 0.3 and cm.prefill_ms_per_visual_token > 0
    assert not DEFAULT.measured
