"""
hitl_demo.py
------------
The bin's decision loop with human-in-the-loop, on a webcam. Everything the bin
does except moving the servo (actuate() only prints, until the main loop and
firmware exist).

For each load:
  1. wait until the plate has settled and nothing new has landed for --wait seconds
  2. assess(): every item labelled by the student, same-compartment check
  3. ReviewPolicy: low confidence -> human; every Xth auto-tilt load -> human (audit)
  4. if a human is asked: KeyboardAsker, then resolve() their labels
  5. FeedbackStore logs the load (and the labelled crops), then actuate()
  6. wait for the plate to be cleared, then start again

Keys in the camera window: r = capture the empty-plate reference, q = quit.

  python hitl_demo.py --student_dir ../student --model_path ../student/runs/bin/weights.pt \
      --feedback ../../feedback --audit_every 20
"""

import argparse
import os
import time

import cv2

from hitl import FeedbackStore, KeyboardAsker, ReviewPolicy, resolve, summarize
from multi_item import (DEFAULT_RULE, DropCounter, ItemDetector, assess, draw, item_crop,
                        load_student, make_classify)


def actuate(action):
    """Stand-in for the servo: replace with the serial command to the ESP32."""
    print(f"  -> {action}")


def load_ready(counter, items, last_drop_time, wait_s, settle_frames):
    return bool(items) and counter.still >= settle_frames and time.time() - last_drop_time >= wait_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bin decision loop with human-in-the-loop (webcam demo)")
    parser.add_argument("--student_dir",     required=True)
    parser.add_argument("--student_module",  default="classifier")
    parser.add_argument("--model_path",      required=True)
    parser.add_argument("--student_version", default=None, help="Name logged with every load (default: weights file name)")
    parser.add_argument("--feedback",        required=True, help="Feedback store folder")
    parser.add_argument("--threshold",       type=float, default=0.6)
    parser.add_argument("--audit_every",     type=int, default=20, help="X: audit every Xth auto-tilt load")
    parser.add_argument("--audit_mode",      choices=["every", "random"], default="every")
    parser.add_argument("--wait",            type=float, default=1.5, help="Seconds after the last drop before deciding")
    parser.add_argument("--camera",          type=int, default=0)
    args = parser.parse_args()

    version = args.student_version or os.path.splitext(os.path.basename(args.model_path))[0]
    classify = make_classify(load_student(args.student_dir, args.model_path, args.student_module))
    detector = ItemDetector()
    counter = DropCounter(detector)
    policy = ReviewPolicy(args.audit_every, args.audit_mode)
    asker = KeyboardAsker()
    store = FeedbackStore(args.feedback, version)
    cap = cv2.VideoCapture(args.camera)

    state, last_drops, last_drop_time, banner = "waiting", 0, time.time(), "clear the plate and press [r]"
    print(f"Logging to {args.feedback} (student {version}, audit every {args.audit_every})")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        shown = frame
        if detector.reference is not None:
            items = detector.detect(frame)
            counter.update(frame, classify)
            if counter.drops != last_drops:
                last_drops, last_drop_time = counter.drops, time.time()

            if state == "waiting" and load_ready(counter, items, last_drop_time, args.wait, counter.settle_frames):
                decision = assess(frame, items, classify, args.threshold, DEFAULT_RULE, counter.drop_items)
                trigger = policy.review(decision)
                human, final = None, decision.action
                if trigger:
                    human = asker.ask([item_crop(frame, it) for it in decision.items],
                                      "audit check" if trigger == "audit" else "not sure")
                    final = resolve(human)
                load_id = store.record(frame, decision, trigger, human, final)
                print(f"{load_id}: {decision.action} ({decision.reason})" + (f", {trigger}: human said {human}" if trigger else ""))
                actuate(final)
                banner = f"{final}  [{decision.composition or '-'}]" + (f"  {trigger}" if trigger else "")
                shown = draw(frame, decision)
                state = "done"
            elif state == "done" and not items:
                counter.reset()                          # plate cleared: next load
                state, last_drops = "waiting", 0
                banner = "ready"
            elif state == "waiting":
                banner = "waiting for items" if not items else "settling..."
        out = shown.copy()
        cv2.rectangle(out, (0, out.shape[0] - 30), (out.shape[1], out.shape[0]), (40, 40, 40), -1)
        cv2.putText(out, f"{banner}   [r] reference  [q] quit", (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("bin", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            detector.set_reference([cap.read()[1] for _ in range(10)])
            counter.reset()
            state, last_drops, banner = "waiting", 0, "ready"

    cap.release()
    cv2.destroyAllWindows()
    if store.n_loads:
        for row in summarize(args.feedback):
            print(row)
