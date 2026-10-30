import csv, os, sys
import cv2, numpy as np
from PIL import Image, ImageDraw
repo, piles, bgdir, th, out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
sys.path.insert(0, os.path.join(repo, "bin_vision")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg import load_refs, segment
from multi_item import item_crop
refs = load_refs(bgdir, th)
rows = list(csv.DictReader(open(os.path.join(piles, "items.csv"), encoding="utf-8")))
res, tiles = [], []
for r in rows:
    frame = cv2.imread(os.path.join(piles, r["file"]))
    it, ref, nb, nd = segment(refs, frame)
    stem = os.path.splitext(os.path.basename(r["file"]))[0]; cpath = ""
    ov = frame.copy()
    if it is not None:
        cpath = f"{r['pile']}/{r['label']} waste/{stem}.jpg"
        os.makedirs(os.path.join(out, os.path.dirname(cpath)), exist_ok=True)
        cv2.imwrite(os.path.join(out, cpath), item_crop(frame, it), [cv2.IMWRITE_JPEG_QUALITY, 95])
        ov[it.mask] = (0.55 * ov[it.mask] + [0, 0, 115]).astype(np.uint8)
        x, y, w, h = it.box; cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 0), 2)
    tiles.append((f"{r['item_id']} {stem[-11:]}", cv2.cvtColor(cv2.resize(ov, (200, 150)), cv2.COLOR_BGR2RGB)))
    res.append({**r, "crop": cpath, "ref": ref, "blobs": nb, "edge_dropped": nd, "area_frac": round(it.area_frac, 4) if it else ""})
with open(os.path.join(out, "crops.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
os.makedirs(os.path.join(out, "check"), exist_ok=True)
C, per = 8, 48
for p in range(0, len(tiles), per):
    ch = tiles[p:p+per]; rw = (len(ch)+C-1)//C; s = Image.new("RGB", (C*200, rw*164), "white"); d = ImageDraw.Draw(s)
    for i, (t, im) in enumerate(ch):
        x, y = (i%C)*200, (i//C)*164; s.paste(Image.fromarray(im), (x, y+14)); d.text((x+2, y), t, fill="black")
    s.save(os.path.join(out, "check", f"sheet_{p//per}.jpg"), quality=85)
print("no item:", [(r["item_id"], r["original"]) for r in res if not r["crop"]])
