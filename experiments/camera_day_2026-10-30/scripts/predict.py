"""predict.py <repo> <crops dir> <pile> <model: student weights.pt | teacher .joblib> <name> <out csv>
One row per cropped photo of the pile: item_id, label, wet_prob, pred, confidence."""
import csv, os, sys
import cv2
repo, cdir, pile, model, name, out = sys.argv[1:7]
if model.endswith(".joblib"):
    sys.path.insert(0, os.path.join(repo, "teacher")); sys.path.insert(0, os.path.join(repo, "student"))
    from teacher import Teacher
    clf = Teacher.load(model)
else:
    sys.path.insert(0, os.path.join(repo, "student"))
    from classifier import WasteClassifier
    clf = WasteClassifier(model_path=model)
rows = [r for r in csv.DictReader(open(os.path.join(cdir, "crops.csv"), encoding="utf-8")) if r["pile"] == pile and r["crop"]]
res = []
for r in rows:
    p = clf.classify_image(cv2.imread(os.path.join(cdir, r["crop"])))
    res.append({"model": name, "crop": r["crop"], "item_id": r["item_id"], "label": r["label"], "wet_prob": round(p["wet_prob"], 5),
                "pred": str(p["label"]).replace(" waste", ""), "confidence": round(p["confidence"], 5)})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(name, pile, len(res), "accuracy", round(sum(x["pred"] == x["label"] for x in res) / len(res), 4))
