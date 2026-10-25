import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="30000")
a = ap.parse_args()
os.environ.setdefault("SGLANG_URL", f"http://localhost:{a.port}/v1")

from factualis.llm_client import chat
from factualis.prompts import build_messages
from factualis.verifier import score_sample

chunk = {
    "doc_id": "demo_kyc",
    "clause": "4.2",
    "text": "4.2 Permanent Account Number (PAN) shall be obtained for any cash transaction of Rs. 50,000 and above. "
    "Transactions above Rs. 10,00,000 require approval of the Principal Officer.",
}
for q, gold in [
    ("When is PAN mandatory for a cash transaction?", "FULLY_GROUNDED"),
    ("What is the AML threshold for crypto staking rewards?", "ABSTAIN_NOT_IN_POLICY"),
]:
    t = time.time()
    raw = chat(build_messages(q, [chunk]))
    s = score_sample(raw, chunk["text"], gold)
    print(f"[{time.time() - t:.2f}s] {q}\n  -> {raw[:300]}\n  status={s.pred_status} gold={gold} errors={s.errors}\n")
