# Controlled General vs Astro Continued Pretraining

This experiment compares two models with the same architecture and training hyperparameters. The only intended variable is the training data.

The goal is to make the previous general-vs-astro sampling comparison cleaner. This is still a tiny experiment, but it is better controlled than comparing checkpoints with different model sizes.

## Controlled Variables

| setting | value |
|---|---:|
| max_iters | 300 |
| eval_interval | 50 |
| eval_iters | 5 |
| batch_size | 4 |
| block_size | 32 |
| n_embd | 64 |
| n_layer | 2 |
| num_heads | 4 |
| num_kv_heads | 2 |
| dropout | 0.1 |
| learning_rate | 3e-4 |
| warmup_iters | 20 |
| lr_decay_iters | 300 |
| tokenizer | GPT-2 BPE |
| params | about 6.57M |

## Compared Runs

| run | data_dir | train tokens | val tokens |
|---|---|---:|---:|
| `compare_general_300` | `data/tiny` | 304222 | 33803 |
| `compare_astro_300` | `data/astro_tiny` | 2243 | 250 |

The two datasets are very different in size. This matters: the astro model can overfit or memorize local patterns much more easily.

## Loss

| step | general train | general val | astro train | astro val |
|---:|---:|---:|---:|---:|
| 0 | 11.0493 | 10.9784 | 11.0235 | 10.9908 |
| 50 | 10.6676 | 10.5837 | 10.5646 | 10.6575 |
| 100 | 9.5149 | 9.5497 | 9.2383 | 9.4647 |
| 150 | 8.4829 | 8.4840 | 7.7700 | 8.1631 |
| 200 | 7.9025 | 7.8213 | 6.9764 | 7.5296 |
| 250 | 7.5013 | 7.6591 | 6.8630 | 7.3034 |

Both models learn. The astro model reaches lower train loss and lower observed val loss by step 250. Because `astro_tiny` is much smaller, this should be interpreted as easier adaptation to a small corpus, not as stronger general capability.

## Sampling Config

```text
temperature = 0.8
top_k = 40
max_new_tokens = 80
device = cpu
```

## Sampling Comparison

| prompt | compare_general_300 | compare_astro_300 | observation |
|---|---|---|---|
| `The ` | `The must thatUT true ... I ... weTh ... not ... shall ...` | `The ... large ... SS often ... precise ... residual ... Earth ... clock ...` | The general model emits generic short tokens and fragments. The astro model emits domain-shaped words such as `precise`, `residual`, `Earth`, and `clock`. |
| `GNSS ` | `GNSS I ... we ... ROM ... you ... know ...` | `GNSS ... temporal ... Inter ... residual ... clock ...` | Under the same model size, the astro model is clearly biased toward geodesy vocabulary. |
| `Space geodesy ` | `Space geodesy will ... In ... I ... shall ...` | `Space geodesy ... measurements ... infrastructure ... links ... domain ...` | The astro model reacts to the in-domain prompt with domain terms, while the general model stays generic. |
| `A terrestrial reference frame ` | `A terrestrial reference frame ... The ... will ... true ...` | `A terrestrial reference frame ... directions ... clock ... ocean ... residual ... infrastructure ... analysis ...` | The astro model associates the prompt with reference-frame-adjacent vocabulary, but the output is still ungrammatical. |
| `RA=` | `RA= ... not ... I ... have ...` | `RA=ge ... estimate ... atmospheric analysis ... remote ... clock ...` | Neither model learns a valid RA/Dec format. The astro model drifts into domain words, not coordinate syntax. |

## Interpretation

This controlled comparison supports the previous qualitative finding: changing the training corpus changes the generation distribution.

The astro run produces more domain-related tokens:

- `GNSS`
- `Earth`
- `clock`
- `precise`
- `residual`
- `measurements`
- `infrastructure`
- `analysis`
- `temporal`
- `remote`

However, the model still does not produce reliable sentences. The outputs contain repeated punctuation, broken subwords, unstable capitalization, and weak grammar. This means the model has mostly learned local token statistics and domain vocabulary, not domain knowledge.

## Lessons

1. Control variables make the comparison easier to interpret.
2. Smaller domain corpora can shift vocabulary quickly, but they also encourage memorization and unstable generation.
3. Continued pretraining on a tiny corpus is useful for observing distribution shift, not for claiming capability.
4. A prompt like `RA=` needs explicit examples of the target format. General domain notes are not enough to teach structured astronomical notation.
5. The next useful improvement is not a larger model, but a larger and better organized astro corpus.

## Next Steps

- Expand `data/raw/astro` toward 50K-100K tokens.
- Separate general-domain and astro-domain eval prompts.
- Add format-specific examples for coordinates, station metadata, and GNSS time series descriptions.
- Repeat the controlled comparison with the same architecture and more data.
