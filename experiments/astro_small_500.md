# Astro Small Continued Pretraining

This experiment expands the astro / geodesy seed corpus from `astro_tiny` to `astro_small`.

The goal is to test whether a larger domain corpus helps the model move from domain-word fragments toward more stable domain-shaped text.

## Corpus

Local raw corpus:

```text
data/raw/astro_small/
```

Prepared data:

```text
data/astro_small/
```

Corpus statistics:

| item | value |
|---|---:|
| input files | 5 |
| chars | 164307 |
| tokens | 41570 |
| chars/token | 3.95 |
| train tokens | 37413 |
| val tokens | 4157 |
| tokenizer | GPT-2 BPE |
| dtype | uint16 |

The corpus contains five source-guided files:

- `space_geodesy_reference_frames.txt`
- `gnss_time_series_and_station_metadata.txt`
- `remote_sensing_insar_timespace.txt`
- `coordinate_formats_and_metadata.txt`
- `domain_explanations_and_questions.txt`

Compared with `astro_tiny`, this corpus intentionally includes more structured examples:

- `RA=12h30m00s, Dec=+45d00m00s`
- `station: BJFS`
- `velocity_east: 2.30 mm/yr`
- `offset_date: 2022-09-18`
- `component: east / north / up`

## Training Config

```bash
python train.py \
  --data-dir data/astro_small \
  --max-iters 500 \
  --eval-interval 50 \
  --eval-iters 10 \
  --batch-size 4 \
  --block-size 32 \
  --n-embd 64 \
  --n-layer 2 \
  --num-heads 4 \
  --num-kv-heads 2 \
  --dropout 0.1 \
  --learning-rate 3e-4 \
  --warmup-iters 30 \
  --lr-decay-iters 500 \
  --out-dir out/astro_small_500
```

Model size: about 6.57M parameters.

## Loss

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 10.9789 | 11.0631 |
| 50 | 10.5423 | 10.7160 |
| 100 | 9.1712 | 9.6157 |
| 150 | 7.3564 | 7.9607 |
| 200 | 6.4343 | 6.9592 |
| 250 | 6.0150 | 6.5719 |
| 300 | 5.5287 | 6.3137 |
| 350 | 5.4891 | 6.2635 |
| 400 | 5.2500 | 6.1563 |
| 450 | 5.0610 | 6.0117 |

Compared with `astro_tiny_300`, the larger corpus produces a smoother and more useful training curve. Validation loss continues to improve through step 450.

## Sampling

Sampling config:

```text
temperature = 0.8
top_k = 40
```

### `GNSS `

Representative output:

```text
GNSS isvel.
station:
station:vel can.
...
offset_date ...
time series ...
```

The model now strongly emits tokens from the station/time-series format. It has learned that GNSS text is associated with `station`, `velocity`, `offset`, and `time series`, but it has not learned to arrange them into a stable record.

### `A terrestrial reference frame `

Representative output:

```text
A terrestrial reference frame ...
coordinate ...
GN deformation ...
metadata ...
residual ...
seasonal ...
offset_east ...
```

The model connects the prompt to coordinate and metadata vocabulary. Sentence structure remains unstable.

### `RA=`

Representative output:

```text
RA=:vel coordinate:
...
Decup:BIstation:
...
RA:
...
```

The model now emits `RA`, `Dec`, `station`, and coordinate-related fragments. However, it still does not generate a valid RA/Dec coordinate format.

### `station: `

Representative output:

```text
station:
velocity
date_ep changes_type
comment:
RA_north_, coordinate: GNSS
```

The model clearly learned station metadata vocabulary, but the fields are not ordered or semantically valid.

## Low-temperature Check

Sampling with:

```text
temperature = 0.5
top_k = 20
```

did not fix the format problem. Instead, the model became more repetitive, producing many `station:`, `_`, `:`, and `vel` fragments.

This suggests the issue is not only sampling randomness. The model has learned high-probability format tokens, but its context modeling is not strong enough to generate coherent structured records.

## Interpretation

The expanded corpus improves domain adaptation:

- lower validation loss than the smaller astro run
- more frequent domain terms
- stronger association between prompts and station / coordinate / metadata tokens

But it still does not produce reliable domain text. The model mostly learns local token statistics and field names.

## Lessons

1. Increasing corpus size from 2.5K to 41.6K tokens improves training stability.
2. Adding format examples makes the model emit format-related tokens.
3. Format tokens are not the same as format competence.
4. Lower temperature cannot repair weak learned structure; it may amplify repetitive high-probability fragments.
5. The next improvement should be better data design, not just more steps.

## Next Steps

- Add cleaner format-specific examples for RA/Dec and station metadata.
- Separate prose data from structured record data.
- Train a simple format-focused corpus and evaluate only format prompts.
- Increase context length after the model can reliably learn short records.
- Consider SFT later for exact structured outputs, because causal LM pretraining alone is weak at instruction-controlled formatting.
