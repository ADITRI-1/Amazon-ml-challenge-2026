# Laya-FR kit — France cross-encoder on map-labelled test data (Neural Ninjas)

Fine-tunes our **Laya v2** cross-encoder (in `model/`, ModernBERT encoder + mean-pool + linear head, Apache-2.0 base
`convaiinnovations/laya-multilingual`) on French pairs from the **test set only** (organiser-approved self-training),
then scores every French test candidate pair. Only `out/scores.parquet` and `out/train_meta.json` come back.

## Data (`data/`, no record IDs)
| file | rows | what |
|---|---|---|
| `train_pairs.parquet` | 1,436,070 | `text_a, text_b, y, src`: 710k real French candidate pairs labelled by the France map (clean copies, legal-form adds and same-address filler variants = 1; same-address category swaps, shifted-number decoys, SNC = 0) + 725k pairs from the map-driven generator (all 259k French S1 anchors). 73% positive |
| `dev_pairs.parquet` | 20,000 | held-out map-labelled real pairs — AUC monitor |
| `score_pairs.parquet` | 1,169,377 | `key, text_a, text_b`: all French test candidate pairs (keys map back only on the main box) |

## Run
```bash
pip install -r requirements.txt
python train_score.py --dtype bf16          # 1 epoch, bs 128, lr 1e-5; prints dev AUC every 2,000 steps
```
Healthy: dev AUC rises (it starts around the Laya v2 level) and loss falls; the script stops on a non-finite loss.
Out of memory → `--bs 64`. `--save_model` also writes the fine-tuned model to `out/model/` (1.2 GB, optional).

## Bring back
`out/scores.parquet` + `out/train_meta.json` → `~/nikhil/experiment/laya_fr_kit/out/` on the main box. Delete the kit
(private data) afterwards.
