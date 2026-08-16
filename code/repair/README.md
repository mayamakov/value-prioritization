# Re-query of failed LLM API calls (August 2026)

A run-log audit found that the original ranking pipeline's parser applied a
batch-level fallback rule: when an API call returned empty or unparsable
content, all 25 pairs in that batch were assigned "first presented patient,
confidence 1" instead of being recorded as failed. This affected:

- Gemini 2.5 Pro: 50 of 300 primary common-pair decisions + 225 rollout decisions
- Qwen3.5 27B:   25 of 300 primary common-pair decisions + 25 rollout decisions (dropped, not fallback-assigned)
- Gemini 3.1 Pro: 75 rollout decisions (dropped, not fallback-assigned)

All affected calls were re-queried in August 2026 through the same OpenRouter
route with the same model identifiers, prompts, and output format
(`rerun_failed_pairs.py` / `rerun_gemini31_standalone.py`, per-pair retries,
no fallback assignment), and the recovered genuine decisions replaced the
fallback-assigned or missing ones in `data/physicians_results/`.
`repair_pairs.json` lists every affected pair in its original presentation order.
32 of the 75 re-queried primary decisions differed from the fallback assignment;
no qualitative conclusion changed (see Supplementary Results 1 of the manuscript).

The only remaining fallback-assigned decisions are 124 training-pair decisions
in Gemini 2.5 Pro's rater-specific active-learning iterations 1-3, which do not
enter any common-pairs analysis. RankNet models were not retrained (their
training data is unchanged), so RankNet-derived artifacts are unchanged.
