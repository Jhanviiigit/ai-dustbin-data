# ai-dustbin-data

Data preparation for the AI dustbin (wet/dry waste) project in
[siansusan/ai_dustbin_2](https://github.com/siansusan/ai_dustbin_2).
This repo holds the cleaning scripts, the reports they produce, and links
to the finished datasets.

## Download the datasets

Use **v2**. It leaves out `dataset2`, whose origin could not be traced.
Datasets are attached to this repo's **Releases** as zip files, not committed to git.

| Dataset | What it is | Images (train / val / test) |
|---|---|---|
| `dataset_C_v2.zip` | Wet/dry **balanced 50/50 within each source** | 2,772 (1,940 / 416 / 416) |
| `dataset_D_v2.zip` | **Background removed** (U²-Net, gray background) + masks | 6,015 (4,214 / 898 / 903) |

```bash
gh release download data-v2 --repo Jhanviiigit/ai-dustbin-data
```

The repo is private: you need to be added as a collaborator. Release `data-v1` is
superseded (it includes dataset2); keep it only for comparison.

Each zip has the same layout, so the existing training script works directly:

```
dataset_C_v2/
  train/  dry waste/  wet waste/
  val/    dry waste/  wet waste/
  test/   dry waste/  wet waste/
  manifest.csv      one row per image: source, original class, duplicate group, split
  report.md         counts per split and source
  DATA_LICENSE.txt  source credits and license
```

```bash
python train_classifier.py --data_dir dataset_C_v2 --batch_size 32
```

Images in the zips are shrunk to at most 512 px (training uses 224) and saved as `.jpg`.
`manifest.csv` still lists each file's original extension, so match on the name
without the extension.

**Never train or tune on `test/`.** Use `val/` for choosing epochs and settings.
`test/` is only for final numbers.

## How the data was built

Sources (v2):

| Source | Images | Notes |
|---|---|---|
| `roboflow` | 1,834 | Roboflow export in ai_dustbin_2 (includes TrashNet-style images). 29% wet. |
| `realwaste` | 4,257 used | [RealWaste](https://github.com/sam-single/realwaste) (Single et al., 2023), real waste photographed at a facility. Food Organics + Vegetation → wet; Cardboard, Glass, Metal, Paper, Plastic, Textile → dry; Miscellaneous Trash left out. 20% wet. |
| ~~`dataset2`~~ | left out | Also in ai_dustbin_2. Origin unknown; it re-used Roboflow and RealWaste photos and some wet images carry stock-photo watermarks. |

Pipeline (`data_prep/`):

1. **`clean_dataset.py`** → `cleaned_v2/` (6,089 images)
   - Tags every image with its source; `--exclude_sources dataset2` leaves dataset2 out.
   - Drops unreadable, tiny or blank images, and exact duplicates.
   - Groups near-duplicates (perceptual hash) and keeps each group inside a single split.
   - Splits 70/15/15, stratified by class **and** source.
2. **`make_balanced.py`** → **dataset C**: downsamples dry so every source is 50/50 wet/dry
   in every split, so the source tells the model nothing about the class.
3. **`remove_background.py`** → **dataset D**: removes backgrounds with rembg (U²-Net,
   Qin et al., 2020), puts the object on flat gray, saves the masks, and drops images
   where the cutout erased the object. About 2.5 hours on a laptop CPU; `--reuse_from`
   reuses masks from an earlier run for identical images.
4. **`package_dataset.py`**: zips a dataset for release (resizes images, adds license credits).

To rebuild from scratch:

```bash
pip install -r requirements.txt
git clone https://github.com/siansusan/ai_dustbin_2.git
git clone https://github.com/sam-single/realwaste.git
python data_prep/clean_dataset.py --src ai_dustbin_2 --realwaste realwaste/RealWaste --exclude_sources dataset2 --out cleaned_v2
python data_prep/make_balanced.py --src cleaned_v2 --out dataset_C_v2
python data_prep/remove_background.py --src cleaned_v2 --out dataset_D_v2
```

Every step uses a fixed seed (42), so the same inputs give the same splits.
On Windows, clone with `git -c core.longpaths=true clone ...` if you hit "Filename too long".

Reports for each version are in `reports/v1/` and `reports/v2/`.

## Known issues

- **Few wet images.** Without dataset2, wet is 23% of the cleaned data (1,386 images), and
  dataset C has only 2,772 images. Train with class weights (`train_classifier.py` does).
- **Background removal fails on piles** (leaf heaps, stacks of pipes) and sometimes on
  transparent items. 74 images (1.2%) were dropped in D v2 because the cutout erased the
  object; Roboflow wet is highest at 5.2%. See `reports/v2/dataset_D_report.md`.
- **No photos from the bin's own camera yet.** These are needed as the real test set.

## License

The data comes from third parties (see `DATA_LICENSE.txt` in each zip). RealWaste is
**CC BY-NC-SA 4.0**, so the combined datasets are non-commercial and must be shared
under the same license.
