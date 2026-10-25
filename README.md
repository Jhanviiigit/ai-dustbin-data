# ai-dustbin-data

Data preparation for the AI dustbin (wet/dry waste) project in
[siansusan/ai_dustbin_2](https://github.com/siansusan/ai_dustbin_2).
This repo holds the cleaning scripts, the reports they produce, and links
to the finished datasets.

## Download the datasets

Datasets are attached to this repo's **Releases** as zip files, not committed to git.

| Dataset | What it is | Images | Status |
|---|---|---|---|
| `dataset_C.zip` | Cleaned data, **wet/dry balanced 50/50 within each source** | 6,034 | Released |
| `dataset_D.zip` | Cleaned data, **background removed** (U²-Net, gray background) + masks | 9,542 | Released |

```bash
gh release download data-v1 --repo Jhanviiigit/ai-dustbin-data --pattern "dataset_C.zip"
gh release download data-v1 --repo Jhanviiigit/ai-dustbin-data --pattern "dataset_D.zip"
```

Each zip has the same layout, so the existing training script works directly:

```
dataset_C/
  train/  dry waste/  wet waste/
  val/    dry waste/  wet waste/
  test/   dry waste/  wet waste/
  manifest.csv      one row per image: source, original class, duplicate group, split
  report.md         counts per split and source
  DATA_LICENSE.txt  source credits and license
```

```bash
python train_classifier.py --data_dir dataset_C --batch_size 32
```

Images in the zips are shrunk to at most 512 px (training uses 224) and saved as `.jpg`.
`manifest.csv` still lists each file's original extension, so match on the name
without the extension.

**Never train or tune on `test/`.** Use `val/` for choosing epochs and settings.
`test/` is only for final numbers.

## How the data was built

Sources:

| Source | Images | Notes |
|---|---|---|
| `roboflow` | 1,834 | Roboflow export in ai_dustbin_2 (includes TrashNet-style images). Mostly dry. |
| `dataset2` | 3,633 | Also in ai_dustbin_2, **origin unknown**. Re-uses some Roboflow and RealWaste photos. Some wet images have stock-photo watermarks. |
| `realwaste` | 4,257 used | [RealWaste](https://github.com/sam-single/realwaste) (Single et al., 2023), real waste photographed at a facility. Food Organics + Vegetation → wet; Cardboard, Glass, Metal, Paper, Plastic, Textile → dry; Miscellaneous Trash left out. |

Pipeline (`data_prep/`):

1. **`clean_dataset.py`** → `cleaned_dataset/` (9,679 images)
   - Tags every image with its source.
   - Drops unreadable, tiny or blank images, and exact duplicates.
   - Groups near-duplicates (perceptual hash) and keeps each group inside a single split.
   - Splits 70/15/15, stratified by class **and** source.
   - What it found: the old ai_dustbin_2 train/val split leaked (58 duplicate groups
     were in both), and 60 duplicate groups span sources. See `reports/cleaned_dataset_report.md`.
2. **`make_balanced.py`** → **dataset C**: downsamples dry so every source is 50/50 wet/dry
   in every split, so the source tells the model nothing about the class.
3. **`remove_background.py`** → **dataset D**: removes backgrounds with rembg (U²-Net,
   Qin et al., 2020), puts the object on flat gray, saves the masks, and drops images
   where the cutout erased the object. Takes about 2.5 hours on a laptop CPU.
4. **`package_dataset.py`**: zips a dataset for release (resizes images, adds license credits).

To rebuild from scratch:

```bash
pip install -r requirements.txt
git clone https://github.com/siansusan/ai_dustbin_2.git
git clone https://github.com/sam-single/realwaste.git
python data_prep/clean_dataset.py --src ai_dustbin_2 --realwaste realwaste/RealWaste --out cleaned_dataset
python data_prep/make_balanced.py --src cleaned_dataset --out dataset_C
python data_prep/remove_background.py --src cleaned_dataset --out dataset_D
```

Every step uses a fixed seed (42), so the same inputs give the same splits.
On Windows, clone with `git -c core.longpaths=true clone ...` if you hit "Filename too long".

## Known issues

- **Where dataset2 came from is unknown.** Ask siansusan; it needs a citation and a license check.
- **Watermarks**: some `dataset2` wet images carry stock-photo watermarks or text overlays.
  Not yet filtered.
- **Background removal fails on piles** (leaf heaps, stacks of pipes) and sometimes on
  transparent items. 137 images (1.4%) were dropped because the cutout erased the object.
  The rate is fairly even, except Roboflow wet at 5.2%. See `reports/dataset_D_report.md`.
- **No photos from the bin's own camera yet.** These are needed as the real test set.

## License

The data comes from third parties (see `DATA_LICENSE.txt` in each zip). RealWaste is
**CC BY-NC-SA 4.0**, so the combined datasets are non-commercial and must be shared
under the same license.
