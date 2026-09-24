"""Aggregate the three GT-drug run JSONs into one summary report."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "gt_eval_output")
DRUGS = ["Tovorafenib", "Lurbinectedin", "Tarlatamab"]

report_lines = []


def w(line=""):
    report_lines.append(line)


grand_bedrock = 0.0
grand_tavily = 0.0
grand_latency = 0.0

for drug in DRUGS:
    d = json.load(open(os.path.join(OUT_DIR, f"{drug}.json"), encoding="utf-8"))
    out = d["output"]
    w(f"=== {drug} ===")
    w(f"status: {d['status']}  total_latency_s: {d['total_latency_s']}  "
      f"total_cost_usd: {d['total_cost_usd']}")
    w(f"comparators found: {[c['generic_name'] for c in out['comparators']]}")
    v = out["validation"]
    w(f"records_extracted={v['records_extracted']} grounded={v['grounded']} "
      f"claim_validated={v['claim_validated']} claim_rejected={v['claim_rejected']} "
      f"subject_drug_rejections={v['subject_drug_rejections']} role_rejections={v['role_rejections']} "
      f"scope_in={v['scope_in']} scope_out={v['scope_out']}")
    w("")
    w("-- per-step latency (s) --")
    for stage, secs in sorted(d["stage_latency_s"].items(),
                              key=lambda kv: -kv[1]):
        w(f"  {stage:10s} {secs:8.2f}")
    w("")
    w("-- Bedrock: per-step calls/tokens/cost --")
    for stage, b in sorted(d["bedrock"]["by_stage"].items()):
        w(f"  {stage:6s} calls={b['calls']:4d} failed={b['failed']:2d} "
          f"in_tok={b['input_tokens']:8d} out_tok={b['output_tokens']:7d} "
          f"cost=${b['cost_usd']:.4f} latency_s={b['latency_s']:.1f}")
    w(f"  TOTAL  calls={d['bedrock']['calls']} failed={d['bedrock']['failed_calls']} "
      f"cost=${d['bedrock']['total_cost_usd']:.4f}")
    w("")
    w("-- Tavily: per-op calls/credits/cost --")
    for op, t in sorted(d["tavily"]["by_op"].items()):
        w(f"  {op:8s} calls={t['calls']:4d} failed={t['failed']:2d} "
          f"credits={t['credits']:.2f} cost=${t['cost_usd']:.4f} latency_s={t['latency_s']:.1f}")
    w(f"  TOTAL    calls={d['tavily']['calls']} failed={d['tavily']['failed_calls']} "
      f"credits={d['tavily']['total_credits']:.2f} cost=${d['tavily']['total_cost_usd']:.4f}")
    w("")
    grand_bedrock += d["bedrock"]["total_cost_usd"]
    grand_tavily += d["tavily"]["total_cost_usd"]
    grand_latency += d["total_latency_s"]

w("=== GRAND TOTAL (3 drugs) ===")
w(f"bedrock_cost_usd={grand_bedrock:.4f}")
w(f"tavily_cost_usd={grand_tavily:.4f}")
w(f"total_cost_usd={grand_bedrock + grand_tavily:.4f}")
w(f"sum_of_run_latencies_s={grand_latency:.1f}")

with open(os.path.join(BASE, "gt_eval_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("wrote", os.path.join(BASE, "gt_eval_summary.txt"))
