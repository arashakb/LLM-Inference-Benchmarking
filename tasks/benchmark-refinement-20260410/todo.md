# To-Do — Benchmark Refinement (2026-04-10)

## Docs

- [ ] Write `docs/batch-size-notes.md` capturing PyTorch vs llama.cpp batching tradeoffs
- [ ] Scaffold `tasks/benchmark-refinement-20260410/` with background, project plan, todo, test report

## Code

### `bench_utils.py`

- [ ] Replace `decode_tokens = max(new_tokens - 1, 1)` with skip-and-warn (Option A)
  - [ ] Track `tpot_skipped` counter
  - [ ] Append to `all_ttft` / `all_latency` / `all_tokens` only when skipping
  - [ ] Add `tpot_skipped` to returned dict
  - [ ] Guard `mean(all_tpot)` / `stdev(all_tpot)` against empty list
- [ ] Update `print_results` to display `tpot_skipped` when > 0
- [ ] Add `format_prompts(tokenizer, prompts) -> list[str]` helper
  - [ ] Returns templated strings if `tokenizer.chat_template` is set
  - [ ] Falls back to raw prompts otherwise

### `benchmark_gguf.py`

- [ ] After loading the quant model, load `quant_tokenizer = AutoTokenizer.from_pretrained(quant_path)`
- [ ] Set `pad_token_id` on `quant_tokenizer` if missing
- [ ] Apply `format_prompts(tokenizer, prompts)` for the FP16 run
- [ ] Apply `format_prompts(quant_tokenizer, prompts)` for the GGUF run
- [ ] Pass `quant_tokenizer` (not base `tokenizer`) into the GGUF `benchmark(...)` call

### `benchmark_quantize.py`

- [ ] Apply `format_prompts(tokenizer, prompts)` for the FP16 run
- [ ] Try-load tokenizer from `quant_id`, fall back to base on failure
- [ ] Apply `format_prompts(quant_tokenizer, prompts)` for the GPTQ run
- [ ] Pass the chosen tokenizer into the GPTQ `benchmark(...)` call

## Verification

- [ ] Static syntax check on each edited file (`python -m py_compile`)
- [ ] (When GPU available) Run `python benchmark_gguf.py`, capture results in `test-report.md`
- [ ] (When GPU available) Run `python benchmark_quantize.py`, capture results in `test-report.md`
- [ ] Confirm DeepSeek-R1-Distill outputs include `<think>` reasoning trace (template applied correctly)
- [ ] Confirm no `tpot_skipped` warnings under normal operation
- [ ] Document TTFT delta vs pre-change baseline in `test-report.md`

## Sign-off

- [ ] All commits land on main
- [ ] Test report filled out
