"""Split the 2026-10-30 bin-camera photos into a test pile and a loop pile, by item.
Groups were read off contact sheets: indexes are positions in the sorted file list of each folder.
A 'source' ties items cut from one physical thing (omelette pieces, one cabbage) to the same pile."""
import csv, glob, os, random, shutil, sys
src_root, out = sys.argv[1], sys.argv[2]
DRY = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],[22,23],[24,25],[26,27],
       [28,29],[30,31],[32,33],[34,35],[36,37],[38,39],[40,41],[42,43],[44,45],[46,47],[48,49],[50,51],[52,53],
       [54,55],[56,57,58],[59,60],[61,62],[63,64],[65,66],[67,68],[69,70],[71,72],[73,74],[75,76],[77,78],[79,80],
       [81,82],[83,84],[85,86,87],[88,89,90],[91],[92,93],[94,95],[96,97],[98,99],[100,101],[102,103],[104,105],
       [106,107],[108,109],[110,111],[112,113,114],[115,116],[117,118,119],[120,121],[122,123],[124,125]]
WET = [[0],[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24],[25,26],[27,28],
       [29,30],[31,32],[33,34],[35,36],[37,38,39],[40,41],[42,43,44],[45,46],[47,48],[49,50],[51,52],[53,54],[55,56],
       [57,58],[59,60],[61,62],[63,64],[65,66],[67,68],[69,70],[71,72],[73,74],[75,76],[77,78],[79,80],[81,82],
       [83,84],[85,86],[87,88,89,90],[91,92],[93,94],[95,96],[97,98],[99,100],[101,102],[103,104],[105,106],
       [107,108],[109],[110,111,112],[113,114]]
SOURCE = {("wet", 1): "omelette", ("wet", 3): "omelette", ("wet", 5): "omelette", ("wet", 7): "omelette",
          ("wet", 9): "chocolate", ("wet", 17): "chocolate", ("wet", 19): "chocolate",
          ("wet", 11): "orange", ("wet", 21): "orange", ("wet", 23): "orange",
          ("wet", 93): "cabbage", ("wet", 95): "cabbage"}
units = []  # (label, [items]); an item = (item_id, [files], source)
for label, groups in (("dry", DRY), ("wet", WET)):
    files = sorted(glob.glob(os.path.join(src_root, label, label, "*.jpg")))
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(files))), f"{label}: groups do not cover each photo exactly once"
    by_source = {}
    for k, g in enumerate(groups):
        item = (f"{label}_{k+1:03d}", [files[i] for i in g], SOURCE.get((label, g[0]), ""))
        if item[2]: by_source.setdefault(item[2], []).append(item)
        else: units.append((label, [item]))
    units += [(label, its) for its in by_source.values()]
rng = random.Random(42)
pile = {}
for label in ("dry", "wet"):
    us = [u for l, u in units if l == label]
    single = [u for u in us if len(u) == 1 and len(u[0][1]) == 1]          # 1 photo: test (test wants 1-2)
    many = [u for u in us if len(u) > 1 or len(u[0][1]) >= 3]              # 3 photos or a multi-item source: loop
    rest = [u for u in us if u not in single and u not in many]; rng.shuffle(rest)
    target = 30
    test = single + rest[:max(0, target - len(single))]
    for u in us:
        for it in u: pile[it[0]] = "test" if u in test else "loop"
if os.path.exists(out): raise SystemExit(f"{out} exists")
rows = []
for label, u in units:
    for item_id, fs, source in u:
        for f in fs:
            dest = os.path.join(out, pile[item_id], f"{label} waste", f"{item_id}__{os.path.basename(f)}")
            os.makedirs(os.path.dirname(dest), exist_ok=True); shutil.copy2(f, dest)
            rows.append({"file": os.path.relpath(dest, out).replace("\\", "/"), "original": f"{label}/{label}/{os.path.basename(f)}",
                         "item_id": item_id, "label": label, "pile": pile[item_id], "source": source, "photos_of_item": len(fs)})
rows.sort(key=lambda r: (r["pile"], r["label"], r["item_id"], r["file"]))
with open(os.path.join(out, "items.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(out, "test_items.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(sorted({r["item_id"] for r in rows if r["pile"] == "test"})) + "\n")
from collections import Counter
items = {r["item_id"]: r for r in rows}
print("items", Counter((r["pile"], r["label"]) for r in items.values()))
print("photos", Counter((r["pile"], r["label"]) for r in rows))
