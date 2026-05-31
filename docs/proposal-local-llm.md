# Proposal — local LLM stage for the categorization cascade

**Status:** draft / for review · captured 2026-05-24
**Owner:** felix.hieber@gmail.com
**Scope:** add an optional on-device LLM stage that replaces (or sits
between) the current `category_similarity` → `zeroshot` tail of the
cascade. Fully local, no cloud calls. Off by default.

This file persists the analysis we did on the Quality tab so the
implementation work can pick up from a settled design instead of
re-deriving it.

---

## 1. Motivation

The Quality tab on the user's real data (1353 user labels, 4-fold CV)
puts the cascade at:

| Stage                   | Coverage | Stage accuracy |
| ----------------------- | -------- | -------------- |
| `vendor_exact_match`    | ~620 rows | 98%            |
| `classifier`            | ~215 rows | 93%            |
| `category_similarity`   | ~220 rows | **22%**        |
| `zeroshot`              | ~185 rows | **9%**         |
| `unknown` (abstain)     | ~100 rows | 0%             |

The first two stages are excellent. The bottom two are net-negative:
each wrong concrete prediction hurts both precision and recall, and on
the 405 hardest rows they collectively contribute fewer correct
predictions than a German-aware LLM should be able to.

**Target.** Replace the `zeroshot` stage with a small local LLM that
actually understands German bank text + the user's category
definitions. Plausible outcome on the existing dataset: 50–70% accuracy
on the 185 rows currently handled by zeroshot, lifting overall
accuracy from 65.3% → ~72%.

---

## 2. Non-goals

- **No cloud APIs.** Hard requirement from the project's privacy
  invariants. The LLM runs on the user's CPU or GPU.
- **No replacement of the fast stages.** `vendor_exact_match`,
  `knn` and `classifier` stay; the LLM never sees a row that the
  classifier confidently labels.
- **No fine-tuning.** Out-of-the-box quantised instruction-tuned
  models, prompted with the user's category list. Keeps the install
  story to "download one GGUF file".

---

## 3. Stack choice

### Runtime: `llama-cpp-python`

| Option                                | Verdict | Why                                                                                                                                      |
| ------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **llama-cpp-python**                  | ✅      | Pure CPU works out of the box; optional cuBLAS/Metal speeds up GPU users. Supports JSON-schema-constrained decoding. Self-contained.     |
| transformers + bitsandbytes           | ❌      | bitsandbytes wants CUDA, larger RAM footprint, competes with sentence-transformers for VRAM. Useful only if user already has CUDA wired. |
| Ollama                                | ❌      | Background server adds an out-of-process dep; harder to ship as a Python package. Same model files.                                      |
| MLX (Apple Silicon only)              | ❌      | Limits portability; llama-cpp's Metal backend is good enough.                                                                            |

New optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
local-llm = [
    "llama-cpp-python>=0.3",  # CPU
    "huggingface_hub>=0.24",  # GGUF download
]
```

For users with NVIDIA, document the env var to build with cuBLAS:

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install --no-cache-dir llama-cpp-python
```

### Default model: `Qwen2.5-7B-Instruct` quantised to `Q4_K_M`

- **Disk:** ~4.4 GB (`qwen2.5-7b-instruct-q4_k_m.gguf`).
- **RAM:** ~5 GB at inference.
- **Speed:** 5–15 tok/s CPU, 50–150 tok/s GPU. ~40 output tokens per
  classification → 1–10 expenses/sec on CPU, 5–40/sec on GPU.
- **Quality:** strong multilingual, excellent instruction following,
  Apache-2.0 license.
- **Source:** `Qwen/Qwen2.5-7B-Instruct-GGUF` on HF hub.

### Alternative models the registry should offer

| Model                                                            | Size (Q4_K_M) | DE quality | Notes                                                                          |
| ---------------------------------------------------------------- | ------------- | ---------- | ------------------------------------------------------------------------------ |
| `Qwen/Qwen2.5-7B-Instruct-GGUF`                                  | 4.4 GB        | strong     | **Default.** Excellent instruction following + multilingual.                   |
| `bartowski/Llama-3.1-8B-Instruct-GGUF`                           | 4.9 GB        | strong     | Meta. Slightly slower than Qwen but very solid German.                         |
| `bartowski/Qwen2.5-3B-Instruct-GGUF`                             | 2.0 GB        | OK         | Half the disk, ~2× speed. Recommended for laptops without GPU.                 |
| `occiglot/occiglot-7b-de-en-instruct`                            | 4.3 GB        | tuned-DE   | DE/EN specifically fine-tuned. Possibly the best ceiling on German bank text.  |
| `DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1-GGUF`            | 4.9 GB        | tuned-DE   | German-fine-tuned Llama-3.

Pick on the same Settings page as the embedding / NLI models. Quantization
levels (Q3, Q4, Q5, Q6) should be selectable; we'd default to Q4_K_M.

---

## 4. Cascade integration

> **Reconciled with the post-batched-predict codebase (2026-05-31).** The
> earlier draft of this section showed a per-row inline hook. That no longer
> matches `classifier.predict_batch`, which since the perf pass runs a
> **two-pass deferred-batch** design. The LLM stage must mirror that shape.

### How `predict_batch` actually works now

`predict_batch` no longer calls the slow fallback per row. It:

1. **Pass 1** — walks every row through stages 1–4 (vendor_exact_match →
   knn → classifier → category_similarity). A row still `low_conf` after
   stage 4 has its premise pushed onto a pending list
   (`zs_pending = [(out_idx, eid, premise)]`) and the loop `continue`s —
   **no model call yet**. Results are written into a pre-sized
   `out: list[Prediction | None]` by `out_idx`, so input order is
   preserved regardless of which pass fills a slot.
2. **Pass 2** — drains `zs_pending` in `cfg.zeroshot.batch_size` chunks
   through `_zeroshot_predict_batch`, writing results back by `out_idx`.

So the LLM stage is **not** an inline `if` in the row loop — it's a
deferred third pass with its own pending list.

### New `STAGE_ORDER`

`evaluation.py`:

```python
STAGE_ORDER = (
    "vendor_exact_match",
    "knn",
    "classifier",
    "category_similarity",
    "llm",         # ← new, before zeroshot
    "zeroshot",
)
```

**Critical, easy-to-miss:** `_apply_stage_mask` in `evaluation.py`
hard-codes each stage's `.enabled` flag (it does **not** loop over
`STAGE_ORDER`). Without an explicit

```python
masked.local_llm.enabled = "llm" in enabled
```

line there, the ablation's cumulative + leave-one-out runs would
silently never toggle the LLM stage — it'd look like the stage doesn't
matter. `_stage_enabled` reads via `getattr(cfg, stage)` and works
generically, **but** note the config section is `local_llm` while the
stage name is `llm`, so `_stage_enabled` needs a name→attr alias (or
name the config section `llm` to match). The `Prediction.stage`
docstring also gains `'llm'`.

### Gating + pass ordering

Same `low_conf` condition the zeroshot pass uses
(`top_conf < cfg.zeroshot.use_when_confidence_below`, or `True` when the
classifier is disabled/absent). Recommended ordering so the
"zeroshot as backstop" behaviour works without re-queuing gymnastics:

1. In Pass 1, route a `low_conf` row to `llm_pending` when
   `local_llm.enabled`; only fall through to `zs_pending` in the same
   row when `local_llm.replaces_zeroshot is False`.
2. Run the **LLM drain pass before the zeroshot drain pass**. Collect
   rows where the LLM returned `(None, 0.0)` (bad/again-unparseable JSON)
   into `zs_pending` *then*, so zeroshot only ever sees the LLM's
   genuine misses. This avoids a second walk over already-answered rows.

```python
# Pass 1 (inside the row loop, replacing the current zeroshot branch):
if low_conf and self.cfg.local_llm.enabled:
    llm_pending.append((i, eid, row_text, vendor_ctx))
    if self.cfg.local_llm.replaces_zeroshot:
        continue
if low_conf and self.cfg.zeroshot.enabled:
    zs_pending.append((i, eid, premise))
    continue

# Between Pass 1 and the existing zeroshot drain:
for (out_idx, eid, conf) , (cid, c) in _drain_llm(llm_pending):
    if cid is not None:
        out[out_idx] = Prediction(eid, cid, c, "llm")
    elif self.cfg.zeroshot.enabled:
        zs_pending.append((out_idx, eid, _premise_for(eid)))  # backstop
    else:
        out[out_idx] = Prediction(eid, None, 0.0, "unknown")
```

The Quality tab ablation ranks the new stage automatically once it's in
`STAGE_ORDER` **and** `_apply_stage_mask` — no manual wiring in the UI.

### Note on "batching"

`llama-cpp-python` has **no true batched inference** for chat
completions — `classify_batch` just iterates while keeping the model
resident in RAM/VRAM. The value of the deferred pass here is (a) one
model load per `predict_batch`, (b) a clean place to tick the progress
callback per row, and (c) a natural spot for the per-text cache (§9).
Don't expect the throughput multiplier that zeroshot's real
`batch_size` gives.

---

## 5. Module layout

```
src/expensa/enrichment/local_llm.py   ← new
src/expensa/features/model_registry.py ← add LLM_MODELS table
src/expensa/config.py                  ← add LocalLLMConfig
src/expensa/ml/classifier.py           ← add stage hook
src/expensa/ui/settings.py             ← add Models > LLM section
tests/unit/test_local_llm.py                    ← canned-output tests
```

### `local_llm.py` skeleton

```python
class LocalLLMClassifier:
    """Lazy-loaded GGUF wrapper. Per-process singleton (the model is
    multi-GB, opening twice is a memory disaster)."""

    def __init__(self, model_path: Path, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx,
                          n_gpu_layers=n_gpu_layers, verbose=False)

    def classify(
        self,
        text: str,
        vendor_context: str,
        categories: list[Category],
        max_tokens: int = 40,
    ) -> tuple[int | None, float]:
        cat_list = "\n".join(
            f"  {i+1}. {c.name}: {c.description}" for i, c in enumerate(categories)
        )
        prompt = (
            "Du klassifizierst eine deutsche Kontotransaktion.\n\n"
            f"Kategorien:\n{cat_list}\n\n"
            f"Transaktion: {text}\n"
            f"Vendor-Kontext: {vendor_context or '(kein Kontext)'}\n\n"
            'Antworte ausschließlich mit JSON: '
            '{"category_number": <int>, "confidence": <float 0-1>}'
        )
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        try:
            parsed = json.loads(out["choices"][0]["message"]["content"])
            cat = categories[int(parsed["category_number"]) - 1]
            return cat.id, float(parsed["confidence"])
        except (KeyError, ValueError, IndexError):
            return None, 0.0
```

### `LocalLLMConfig` (pydantic)

```python
class LocalLLMConfig(BaseModel):
    enabled: bool = False               # off by default
    model_path: str = ""                # absolute path to .gguf
    n_ctx: int = 4096
    # -1 offloads all layers to the GPU. The maintainer's box has an
    # RTX 5090, so -1 is the sensible default here; CPU-only installs
    # should set 0.
    n_gpu_layers: int = -1
    max_tokens: int = 40
    temperature: float = 0.0            # 0.0 keeps results deterministic
    # Replace zeroshot for the abstention bucket (default), or run as an
    # independent stage with zeroshot still active behind it.
    replaces_zeroshot: bool = True
```

> The cascade hook is no longer an inline per-row call — see §4 for the
> deferred-batch integration (`llm_pending` drained between Pass 1 and
> the zeroshot drain). The `LocalLLMClassifier` exposes
> `classify_batch(items) -> list[(cid, conf)]` rather than a single-row
> `classify`, matching how the zeroshot path was refactored.

---

## 6. Tradeoffs vs current zero-shot NLI

| Dimension              | mDeBERTa NLI | Local 7B LLM (Q4)              |
| ---------------------- | ------------ | ------------------------------ |
| Accuracy on user data  | ~9%          | **50–70%** (target)            |
| Latency / row          | 10–50 ms     | 500 ms–5 s (CPU) / 50–200 ms (GPU) |
| Disk                   | 800 MB       | 4.4 GB                         |
| RAM at inference       | ~1.5 GB      | 5–6 GB                         |
| First-load time        | ~3 s         | 5–10 s                         |
| Privacy                | local        | local                          |
| Determinism            | yes          | yes at `temperature=0`         |

For the user's 1353-row dataset, a full `Predict-all` run rises from
~30 s → ~15 min on CPU (or ~2 min on GPU). Acceptable to run nightly.

---

## 7. Settings UI

Add a third expander under Models — call it **Local LLM** — with the
same model-table-and-picker pattern as Embeddings / Zero-Shot, plus
a separate form for the prompting/runtime knobs (n_ctx, n_gpu_layers,
max_tokens, temperature, replaces_zeroshot toggle).

Also surface `cfg.local_llm.enabled` in the cascade tuning section
alongside the other stage toggles so users can A/B it in the Quality
tab. Add `"llm"` to the ablation `STAGE_ORDER` so the cumulative + LOO
charts automatically include it.

---

## 8. Testing strategy

- **Unit tests:** stub `LocalLLMClassifier.classify` to return canned
  predictions (no actual GGUF load in unit tests — the dep is optional
  and a 4 GB download is not appropriate for CI).
- **Integration / smoke:** marked `@pytest.mark.slow`, skipped unless
  the env var `EXPENSA_LLM_GGUF_PATH` points at a real GGUF on disk.
- **Determinism gate:** assert two consecutive calls with
  `temperature=0` produce identical outputs.
- **Failure modes:** test that a malformed LLM response (non-JSON,
  out-of-range category index, missing key) returns `(None, 0.0)`
  rather than crashing the cascade.

---

## 9. Caching

Per-text in-memory LRU on `(text, vendor_context)` hash keys so the
same Predict-all run doesn't re-prompt the LLM for duplicate strings
(common with recurring vendors). Optional on-disk cache table
`llm_predictions(text_hash, model_id, category_id, confidence,
predicted_at)` if rerun cost becomes painful.

---

## 10. Open questions

1. **Default off vs default on once it lands.** Recommend off by
   default with a one-click "Try it" CTA in the Quality tab summary
   panel. The 4 GB download is a real cost.
2. **Model picker friction.** Should we ship a "recommended quantise"
   pre-baked into the registry, or always let users pick the
   quantisation level? Lean toward shipping `Q4_K_M` only for the
   default model, full menu for power users.
3. **Replace vs. supplement zeroshot.** `replaces_zeroshot=True` is
   the cleanest default since zeroshot loses every measured A/B once
   the LLM is on. Keep `zeroshot` available for users without enough
   RAM/disk for the LLM.
4. **Fallback when JSON parsing fails.** Two reasonable answers: (a)
   abstain (current draft), (b) try `temperature=0.3` once and re-parse.
   Lean toward (a) for determinism.

---

## 11. Rollout plan

Land in a separate branch (`feat/local-llm-stage`) since:

- Adds a non-trivial optional dep (`llama-cpp-python` requires a C
  compiler on install for CPU; users without one need a prebuilt wheel).
- Requires a 4 GB model download to actually try, gated behind the
  user clicking "Download" in Settings.
- Touches `STAGE_ORDER`, which changes the ablation contract — every
  cached eval result becomes stale (the eval cache's schema-version
  bump handles this gracefully but it's still a one-time UX nick).

Sequence inside the branch:

1. `feat(config): LocalLLMConfig`
2. `feat(enrichment): LocalLLMClassifier wrapper + JSON-schema prompt`
3. `feat(registry): LLM_MODELS table + download helpers`
4. `feat(cascade): wire 'llm' stage in classifier.predict_batch`
5. `feat(ui): Models > Local LLM picker + prompting form`
6. `feat(eval): bump STAGE_ORDER + cache schema_version`
7. `docs: README + CLAUDE.md notes; mark this proposal as IMPLEMENTED`
