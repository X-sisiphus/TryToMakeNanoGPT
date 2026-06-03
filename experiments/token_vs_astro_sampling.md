# Token vs Astro Sampling

This note compares two small tokenizer-level GPT checkpoints:

- `out/token_300/ckpt.pt`: trained on `data/tiny`
- `out/astro_tiny_300/ckpt.pt`: trained on `data/astro_tiny`

The goal is not to prove model capability. The goal is to observe how training data changes the generation distribution.

Important caveat: this is not a strictly fair comparison. The two models use different data and slightly different model sizes. The comparison is still useful as a qualitative learning experiment.

## Setup

| item | token_300 | astro_tiny_300 |
|---|---:|---:|
| data | `data/tiny` | `data/astro_tiny` |
| data type | general tiny text | astro / geodesy seed corpus |
| params | about 9.90M | about 6.57M |
| steps | 300 | 300 |
| tokenizer | GPT-2 BPE | GPT-2 BPE |
| final observed train loss | 6.3929 | 6.8630 |
| final observed val loss | 6.5479 | 7.3034 |

## Sampling Config

```text
temperature = 0.8
top_k = 40
max_new_tokens = 80
device = cpu
```

## Results

| prompt | token_300 sample | astro_tiny_300 sample | observation |
|---|---|---|---|
| `The ` | `The , the. ... I ... be, ... to my my the to you` | `The y,, and-, de and and https de... remote, clock... Earth... measurements... residual...` | The general model mostly emits function words, punctuation, and short fragments. The astro model is still broken, but domain words such as `remote`, `clock`, `Earth`, `measurements`, and `residual` appear. |
| `GNSS ` | `GNSS 's in the ... and of ... this ... your have ...` | `GNSS A ... temporal ... Earth ... precise scale ... Inter measurements ... estimate ... ocean ...` | The astro model shifts toward geodesy vocabulary: `temporal`, `Earth`, `precise`, `measurements`, `estimate`, `ocean`. |
| `Space geodesy ` | `Space geodesy in and, is have ... your my ... this ...` | `Space geodesy ... measurements ... Earth ... precise ... satellite ... gov ...` | With an in-domain prompt, the astro model more often continues with domain-related terms, although syntax remains unstable. |
| `A terrestrial reference frame ` | `A terrestrial reference frame ... that I ... of ... The of ... him ...` | `A terrestrial reference frame ... clock frame ... infrastructure ... Earth ... satellite ... concepts ...` | The astro model associates the prompt with `clock`, `frame`, `infrastructure`, `Earth`, `satellite`, and `concepts`; the general model does not. |
| `RA=` | `RA= the ... him the my the ... to be ...` | `RA= anddata ... Earth ... remote Space ... residual ... clock ... estimate ... links ...` | Neither model learns a valid RA/Dec coordinate format. The astro model still drifts toward domain words rather than coordinate syntax. |

## Interpretation

The most visible effect of `astro_tiny` continued pretraining is vocabulary shift. The model starts to emit words that were frequent in the astro / geodesy seed corpus:

- `Earth`
- `GNSS`
- `clock`
- `satellite`
- `temporal`
- `measurements`
- `estimate`
- `infrastructure`
- `residual`
- `remote`

However, the samples are not semantically reliable. They contain repeated punctuation, broken words, awkward fragments, and weak sentence structure. This suggests the model has learned local token statistics rather than robust domain reasoning.

## Lessons

1. Loss alone is not enough. Both models reduce loss, but sampling reveals what kind of distribution each model has learned.
2. Small-domain continued pretraining can quickly bias generation toward domain vocabulary.
3. Domain vocabulary shift is not the same as domain capability.
4. The `astro_tiny` corpus is too small for reliable language generation. It is useful for pipeline validation and qualitative observation, not for capability claims.
5. Prompt `RA=` shows a limitation: the model does not learn astronomical coordinate format from this seed corpus. More targeted data would be needed.

## Next Questions

- What happens if the astro corpus is expanded from 2.5K tokens to 50K-100K tokens?
- Does a same-size model trained on general text vs astro text show the same vocabulary shift?
- Can prompt-specific examples teach formats such as `RA=12h30m, Dec=+45deg`?
- Would a Chinese-friendly tokenizer improve training efficiency for Chinese astronomy / time-space intelligence texts?
