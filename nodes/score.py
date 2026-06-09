def score_node(state):
    verdicts = state["verdicts"]

    if not verdicts:
        return {"score": 0.0}

    total_claims = len(verdicts)

    # Weights for each verdict type
    VERDICT_WEIGHTS = {"supported": 1.0, "unverifiable": 0.5, "contradicted": 0.0}

    weighted_sum = 0.0

    for claim, data in verdicts.items():
        verdict = data.get("verdict", "unverifiable")
        confidence = data.get("confidence", 0.0)
        base = VERDICT_WEIGHTS.get(verdict, 0.5)

        # Blend base weight with confidence
        # e.g. supported but confidence 0.5 → 0.75, not a full 1.0
        weighted_sum += (base + confidence) / 2

    score = round(weighted_sum / total_claims, 2)
    return {"score": score}
