# FlowC&alpha;: Accurate C&alpha; Atom Prediction in Cryo-EM Maps via Flow Matching


FlowC&alpha is a flow‑matching‑based framework that directly predicts the continuous spatial distribution of C&alpha coordinates from cryo‑EM density maps.

## 📦 Installation

```bash
git clone https://github.com/zhtianz/FlowCa.git
cd FlowCa
```

## 🚀 Usage

### Basic inference command

```bash
python inference.py \
    --map_path /path/to/density.mrc \
    --model_path /path/to/pretrained_model.pt \
    --output_dir ./results
```

### Full options

| Argument | Description |
|----------|-------------|
| `--device` | Device to use (`cuda` or `cpu`, default: `cuda` if available) |
| `--batch_size` | Batch size per GPU (effective batch size = batch_size * accum_iter * #gpus) |
| `--input_size` | Input patch size (pixels) for density map crops |
| `--model_path` | Pretrained FlowCα model checkpoint path |
| `--map_path` | Path to input density map file (`.mrc` / `.map` format) |
| `--contour` | Recommended contour level for density map (used for visualization) |
| `--threshold` | Probability threshold for Cα prediction (default: 0.5) |

### Example

```bash
python inference.py \
    --device cuda \
    --batch_size 4 \
    --input_size 64 \
    --model_path checkpoint.pth \
    --map_path ./examples/emd_26993.mrc \
    --contour 0.17 \
    --threshold 0.5
```
---

## 📊 Output

All output files are saved in the **same directory as the input density map** (i.e., the folder containing `--map_path`). Two CIF files are generated:

| File Name | Description |
|-----------|-------------|
| `output_ca_points_before_pruning.cif` | Raw predicted C&alpha coordinates **before** clustering. This file contains all candidates directly sampled from the flow matching model and may include **false positives** (extra atoms) due to noise or ambiguous density. |
| `see_alpha_output_ca.cif` | **Final predicted C&alpha atoms after clustering and pruning.** This is the recommended result to use for downstream structure analysis or model building. Clustering removes spurious points and keeps only high‑confidence C&alpha positions. |

> 💡 **Tip:** The `--threshold` parameter controls the probability cutoff during inference, which affects both raw predictions and the final clustering result.
