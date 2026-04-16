# Perplexity Calculation

## What perplexity measures

Perplexity answers: "how surprised is the model by this text?" At each position, the model predicts a probability distribution over the next token. If the model assigns high probability to the actual next token, that's low surprise. Perplexity is the exponential of the average surprise (cross-entropy loss) across all positions:

```
PPL = exp(average negative log-likelihood per token)
```

**PPL = 1** means perfect prediction. **PPL = 12.57** (our FP16 result) means the model is, on average, as uncertain as if choosing uniformly among ~12.6 equally likely tokens.

## The sliding window approach

WikiText-2 test is ~299K tokens, but the model's context window is limited (2048 in our config). We can't feed the whole text at once. Instead, we slide a window across the text:

```
Tokens: [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, ...]

max_length = 6,  stride = 3

Window 1: [t0, t1, t2, t3, t4, t5]     <- score ALL tokens (first window)
Window 2:          [t3, t4, t5, t6, t7, t8]     <- score only t6, t7, t8
Window 3:                   [t6, t7, t8, t9, t10, t11]  <- score only t9, t10, t11
```

Each window has **overlap** -- the previous `max_length - stride` tokens serve as context but aren't scored again. This is the masking logic in `bench_utils.py`.

## How the masking works (PyTorch version)

Traced through `bench_utils.py:compute_perplexity()` with concrete numbers.

Say `max_length=8`, `stride=4`, and we have 20 tokens:

**Window 1** (`begin=0`, `end=8`):
```
chunk:  [t0, t1, t2, t3, t4, t5, t6, t7]
target: [t0, t1, t2, t3, t4, t5, t6, t7]   <- all scored (begin == 0)
```
The model's cross-entropy loss is computed over all 8 positions. HuggingFace internally shifts logits vs labels (logits at position `i` predict `target[i+1]`), so effectively 7 token predictions are scored.

**Window 2** (`begin=4`, `end=12`):
```
chunk:  [t4,  t5,  t6,  t7,  t8,  t9, t10, t11]
target: [-100,-100,-100,-100, t8,  t9, t10, t11]
         ^^^^ overlap context ^^^^  ^^^^ scored ^^^^
```
Here `num_keep = min(stride, end - begin) = min(4, 8) = 4`, so we mask the first `8 - 4 = 4` positions with `-100`. HuggingFace ignores `-100` labels in loss computation. Tokens t4-t7 provide context (the model "sees" them) but their predictions don't count toward the loss -- they were already scored in Window 1.

**Window 3** (`begin=8`, `end=16`): same pattern, scores t12-t15 with t8-t11 as context.

**Final window** (`begin=12`, `end=20`, only 8 tokens left): scores t16-t19, breaks loop because `end == seq_len`.

## How the MLX version differs

In `bench_utils.py:compute_perplexity_mlx()`, MLX models don't accept `labels=`, so we do the shift manually:

```python
logits = model(chunk[None])        # shape: (1, chunk_len, vocab_size)
shift_logits = logits[:, :-1, :]   # predict positions 1..end
shift_labels = chunk[1:]           # actual tokens at positions 1..end
```

Then we slice to keep only the stride portion:
```python
# For window 2 with max_length=8, stride=4:
score_start = max_length - stride - 1  # = 8 - 4 - 1 = 3
score_logits = shift_logits[:, 3:, :]  # last 4 predictions
score_labels = shift_labels[3:]        # last 4 target tokens
```

The `-1` accounts for the shift -- in the shifted sequence, position 3 corresponds to predicting original token at index 4 (the start of the non-overlap region).

## Putting it all together

After processing all windows:
```python
# Each window contributes (mean_loss, num_tokens_scored)
total_nll = sum(loss * count for loss, count in nlls)
total_tokens = sum(count for _, count in nlls)
avg_nll = total_nll / total_tokens
perplexity = math.exp(avg_nll)  # e.g., exp(2.53) ~ 12.57
```

The weighted average ensures short final windows don't skew the result.

## Why overlap matters

Without overlap, each window would start "cold" -- the model has no context for the first tokens and makes poor predictions, artificially inflating perplexity. The overlap gives the model `max_length - stride = 1536` tokens of context before scoring new tokens, closely matching how the model would process continuous text in practice.
