# Preparing the Adobe-DPE dataset

This guide covers building the **Adobe-DPE** dataset used to train and evaluate
DeepLPF: the MIT-Adobe FiveK photographs, processed through Adobe Lightroom
following the Deep Photo Enhancer (DPE) protocol, with Expert C retouching as
the target. It is the path that reproduces the paper's reported numbers.

There is an easier alternative — the 480p Expert-C FiveK hosted by the
[Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) repo —
but it uses a different preprocessing and split, so it will not reproduce
DeepLPF's Adobe-DPE numbers exactly. Use it if you only need a standard FiveK
benchmark; use the steps below if you want faithfulness to the paper.

## What you need

- The **MIT-Adobe FiveK** dataset (~50 GB of raw DNG files plus a Lightroom
  catalogue): <https://data.csail.mit.edu/graphics/fivek/>
- **Adobe Lightroom Classic** (the catalogue `fivek.lrcat` opens in Lightroom).
- **Disk space:** the FiveK archive is ~47 GiB and extracts to a similar size, so
  plan for **~95 GiB free** for the raw dataset (an external drive is fine), plus
  a few GB for the exported 512px PNGs.

The raw download and the Lightroom export are manual steps — Lightroom is a GUI
application and cannot be scripted here. The scripts in `data_prep/` handle
everything after the export.

## Step 1 — Download and open the catalogue

1. Download and extract FiveK from the link above (the single 50 GB archive or
   the multi-part version).
2. Open `fivek.lrcat` in Lightroom Classic and let any catalogue upgrade finish
   (this can take several minutes).

In the **Collections** panel you will see an `Inputs` group (several
white-balance renderings of the raw files) and an `Experts` group (`A`–`E`, the
five retouchers). DeepLPF uses an input rendering as the **input** and **Expert
C** as the **target**.

## Step 2 — Export the input images

1. In **Collections**, select `InputAsShotZeroed`. This is the rendering to use:
   exports from it reproduce this repo's bundled reference inputs exactly
   (`mean|Δ| = 0.00` on 8 of the 10 reference images, the exceptions being
   `a4514` and `a4783` at around 9-10/255), and it is the collection
   named by the [FiveK-with-Lightroom guide](https://github.com/yuanming-hu/exposure/wiki/Preparing-data-for-the-MIT-Adobe-FiveK-Dataset-with-Lightroom).
   Note that DPE's own README does not state which input collection it used, so
   this is established by matching the reference images rather than by citation.

   Earlier versions of this document recommended
   `Inputs/Input with Daylight WhiteBalance minus 1.5`. That is **wrong** — the
   `minus 1.5` is a −1.5 EV exposure cut, and it yields inputs roughly
   1.6× too dark, failing the Step 5 check with `mean|Δ|` around 20–60.
2. Select all images (`Ctrl`/`Cmd`-`A`), right-click, **Export**.
3. Export settings:
   - **Format:** PNG
   - **Colour space:** sRGB
   - **Bit depth:** 8 bits/component
   - **Image sizing:** resize so the **long edge = 512 px**, do not enlarge
   - **Destination:** an empty folder, e.g. `~/fivek_export/input/`

The 512 px long edge matches the DeepLPF example images (which are 512×341 /
512×343). Lightroom preserves the original filenames (`aXXXX-<camera>.png`), and
the `aXXXX-` prefix is the image id the data loader keys on — keep it.

## Step 3 — Export the Expert C targets

1. In **Collections**, select `Experts/C`.
2. Select all, right-click, **Export**, with the **same settings as Step 2**
   (PNG, sRGB, 8-bit, long edge 512) to a second folder, e.g.
   `~/fivek_export/output/`.

## Step 4 — Organise into the expected layout

The data loader expects a directory with an `input/` folder and an `output/`
folder, each holding files named `<id>-....png`. Either export directly into
those two folders, or run:

```bash
python data_prep/organise_fivek.py ~/fivek_export/input ~/fivek_export/output \
    ./adobe5k_dpe_data --long-edge 512
```

This converts each image to RGB (dropping any alpha), resizes to a 512 px long
edge, pairs input and target by id, and writes:

```
adobe5k_dpe_data/
  input/   a0001-....png ...
  output/  a0001-....png ...
```

Pass `--no-resize` if you exported at the target resolution already.

## Step 5 — Verify before training

```bash
python data_prep/verify_dataset.py ./adobe5k_dpe_data
```

This runs the repo's own `Adobe5kDataLoader` over the `train`, `valid` and
`test` splits in `adobe5k_dpe/` and reports how many ids were paired, which are
missing, and the channel count of a sample. It also compares your exported
inputs against the bundled example images
(`adobe5k_dpe/deeplpf_example_test_input/`) — a small mean difference means your
preprocessing is consistent with DeepLPF's; a large one means the input develop
settings differ and Step 2 needs revisiting.

Expected split sizes are 2250 train / 2250 valid / 500 test. Note the splits in
`adobe5k_dpe/*.txt` are a best guess: as `adobe5k_dpe/readme.txt` records, the
original DPE splits were unavailable when this repo was written, so exact
reproduction of the paper's numbers may differ slightly from the split alone.

## Step 6 — Train and evaluate

Train (batch size 1 reproduces the paper; higher batch sizes are supported for
throughput):

```bash
python main.py \
  --training_img_dirpath=./adobe5k_dpe_data/ \
  --train_img_list_path=./adobe5k_dpe/images_train.txt \
  --valid_img_list_path=./adobe5k_dpe/images_valid.txt \
  --test_img_list_path=./adobe5k_dpe/images_test.txt \
  --batch_size=1
```

**Batch size > 1** needs a uniform image size, because FiveK images vary in
dimensions and the default collation cannot stack different-sized tensors. Pass
`--crop_size` to randomly crop training pairs to a fixed square, e.g. for a batch
of 8 with 256×256 crops:

```bash
python main.py \
  --training_img_dirpath=./adobe5k_dpe_data/ \
  --train_img_list_path=./adobe5k_dpe/images_train.txt \
  --valid_img_list_path=./adobe5k_dpe/images_valid.txt \
  --test_img_list_path=./adobe5k_dpe/images_test.txt \
  --batch_size=8 --crop_size=256
```

Evaluation and inference run at batch size 1 and are not cropped.

Run inference with a pretrained checkpoint:

```bash
python main.py \
  --inference_img_dirpath=./adobe5k_dpe/ \
  --inference_img_list_path=./adobe5k_dpe/images_inference.txt \
  --checkpoint_filepath=./pretrained_models/adobe_dpe/deeplpf_psnr_24.180_ssim_0.917_model.pt
```

## Notes on faithfulness

- The pretrained `adobe_dpe` checkpoint gets close to, but does not exactly
  reproduce, the 2020 per-image PSNRs recorded in the bundled example
  filenames. Re-running the five `TEST_425` examples on CPU under the current
  code gives 35.84 / 19.16 / 29.32 / 26.83 / 21.87 dB against filename claims
  of 34.60 / 18.94 / 28.00 / 29.83 / 24.23 - agreeing within about 1.3 dB on
  three and differing by 3.0 dB on `a4742` and 2.4 dB on `a4774`. Treat the
  filenames as a 2020 record, not as a target to hit. Running the checkpoint
  over your full prepared test split is the Tier-2 check that the end-to-end
  pipeline matches the paper's 23.90 dB / 0.911 SSIM.
- Results can drift slightly from the 2020 paper for two reasons independent of
  your data: the best-guess splits above, and PyTorch version differences
  (the paper used 1.7.1). Both are usually small.

## Sources

- MIT-Adobe FiveK: <https://data.csail.mit.edu/graphics/fivek/>
- FiveK preparation with Lightroom: <https://github.com/yuanming-hu/exposure/wiki/Preparing-data-for-the-MIT-Adobe-FiveK-Dataset-with-Lightroom>
- Deep Photo Enhancer (DPE): <https://github.com/nothinglo/Deep-Photo-Enhancer>
