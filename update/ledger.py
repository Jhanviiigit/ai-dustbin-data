"""
ledger.py
---------
Remembers which feedback rows are already in the pool, when the last update
was, and how many updates there have been. Lives at pool/ledger.json.

Feedback rows are only marked processed when an update commits them; if the
trigger says "not yet", they stay pending and are picked up next time.
"""

import json
import os
from datetime import datetime


class Ledger:
    def __init__(self, pool_root):
        self.path = os.path.join(pool_root, "ledger.json")
        data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        self.processed = set(data.get("processed", []))
        self.updates = int(data.get("updates", 0))
        self.started = data.get("started") or datetime.now().isoformat(timespec="seconds")
        self.last_update = data.get("last_update")

    def since_last_update(self, now=None):
        """Days since the last committed update (or since the ledger was started)."""
        now = now or datetime.now()
        return (now - datetime.fromisoformat(self.last_update or self.started)).total_seconds() / 86400

    def commit(self, crop_ids, now=None):
        """Mark these feedback rows processed and count one more update. Returns its number."""
        self.processed.update(crop_ids)
        self.updates += 1
        self.last_update = (now or datetime.now()).isoformat(timespec="seconds")
        self.save()
        return self.updates

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"started": self.started, "last_update": self.last_update, "updates": self.updates,
                       "processed": sorted(self.processed)}, f, indent=1)
        os.replace(tmp, self.path)
