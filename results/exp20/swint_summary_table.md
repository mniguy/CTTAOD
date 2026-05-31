# Exp20 Swin-T COCO -> COCO-C Summary

WHW `Avg.` is treated as AP16, including `Org.`.

## Completed Runs

| Rank | Tag | Method | AP15 | AP16 | Δ WHW Ours | Back imgs | FPS | Noise Δ | Digital Δ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `coco_c_swint_solB_ewc` | Sol-B + EWC | 21.10 | 22.44 | -0.20 | 80000 | 6.91 | -0.45 | -0.67 |

## WHW Swin-T Reference

| Method | AP15 | AP16 | Back imgs | FPS |
|---|---:|---:|---:|---:|
| Direct-Test | 15.97 | 17.66 | 0 | 21.5 |
| Ours | 21.33 | 22.64 | 80000 | 9.5 |
| Ours-Skip | 20.31 | 21.62 | 9700 | 17.7 |

## Per-Domain AP

| Tag | Gau | Sht | Imp | Def | Gls | Mtn | Zm | Snw | Frs | Fog | Brt | Cnt | Els | Px | Jpg | Org |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coco_c_swint_solB_ewc` | 13.0 | 16.0 | 16.0 | 14.1 | 13.5 | 14.4 | 8.3 | 23.9 | 27.2 | 37.3 | 36.6 | 26.3 | 27.4 | 20.6 | 21.9 | 42.7 |
