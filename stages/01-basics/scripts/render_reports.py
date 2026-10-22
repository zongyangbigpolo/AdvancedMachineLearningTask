"""Render concise screenshot tables from saved measurements, without training."""

import argparse
import json
from pathlib import Path

from aml.recording import save_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("02-systems", "03-data-scaling", "04-alignment", "05-agents"), required=True)
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[2] / args.stage / "reports"
    result = json.loads((report / "results.json").read_text())
    if args.stage == "02-systems":
        sections = {
            "Attention: synchronized forward + backward, FP32, MPS": [
                {"tokens": row["length"], "reference_ms": round(row["reference_forward_backward"]["median_ms"], 4),
                 "SDPA_ms": round(row["torch_sdpa_forward_backward"]["median_ms"], 4),
                 "speedup": round(row["speedup"], 4), "forward_error": row["forward_max_abs_error"],
                 "gradient_error": row["backward_max_abs_error"]}
                for row in result["attention"]
            ],
            "Greedy decoding: identical outputs, 32-token prompt": [
                {"new_tokens": row["new_tokens"], "uncached_ms": round(row["uncached"]["median_ms"], 3),
                 "cached_ms": round(row["cached"]["median_ms"], 3), "speedup": round(row["speedup"], 4),
                 "cache_bytes": row["cache_bytes"], "identical_output": row["identical_output"]}
                for row in result["generation"]
            ],
            "Unmeasured hardware tasks": result["hardware_blocked"],
        }
    elif args.stage == "03-data-scaling":
        card = result["data"]
        sections = {
            "Raw XML to cleaned, disjoint training and validation": {
                key: card[key] for key in ("source", "raw_sha256", "raw_bytes", "license", "format",
                                           "input_documents", "kept_documents", "removed",
                                           "train_documents", "validation_documents", "cross_split_exact_duplicates")
            },
            "Actual fixed-budget training, three seeds per size": [
                {key: run[key] for key in ("parameters", "seed", "tokens_seen", "validation_ce")}
                for run in result["runs"]
            ],
            "Prediction made without the largest model": result["extrapolation"],
        }
    elif args.stage == "04-alignment":
        sections = {
            "Same-base, three-seed held-out comparison": result["comparison"],
            "Examples from every method, seed 42": [
                {"method": run["method"], **example}
                for run in result["runs"] if run["seed"] == 42
                for example in run["test"]["examples"][:5]
            ],
            "Objective definitions and fair-budget protocol": {
                key: result[key] for key in ("task", "policy", "base_sha256", "protocol", "sft", "dpo", "rlvr")
            },
        }
    else:
        traces = json.loads((report / "traces.json").read_text())
        episode = next(row for row in traces if row["label"] == "plus_self_judge" and row["goals"] == 8)
        sections = {
            "Actual task training and external reward": {
                key: value for key, value in result["training"].items() if key != "curve"
            },
            "Held-out complete-episode success and model-call cost": [
                {key: row[key] for key in ("label", "goals", "episodes", "verified_successes", "mean_model_calls")}
                for row in result["evaluation"]
            ],
            "Actual neural-agent trajectory, first and last four actions": {
                "verified_success": episode["verified_success"], "steps": episode["steps"],
                "targets": episode["targets"], "final_values": episode["final_values"],
                "first_actions": episode["trace"][:4], "last_actions": episode["trace"][-4:],
            },
        }
    save_record(report, f"Measured experiment: {args.stage}", result, sections)


if __name__ == "__main__":
    main()
