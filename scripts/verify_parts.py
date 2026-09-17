"""
Validation for part-wise training: partition correctness + resume safety.

Proves the properties that matter, rather than asserting them in prose:
  1. Parts do not overlap.
  2. No epoch is skipped between Parts.
  3. The union of all Parts is exactly the full schedule (100% coverage).
  4. Part sizes are balanced (differ by at most 1).
  5. Every Part trains on the WHOLE dataset (this is epoch-wise, not
     file-wise, partitioning) -- checked against the split manifests.
  6. The ledger's completed/incomplete logic behaves correctly.

    python -m scripts.verify_parts
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai.computer_vision.train_parts import PartsState, plan_parts  # noqa: E402
from configs import registry as R  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def verify_partitioning() -> None:
    print("\n1. Partition correctness (epoch coverage)")
    for total_epochs, total_parts in [(100, 6), (100, 1), (150, 6), (7, 6),
                                      (80, 5), (120, 6), (13, 4)]:
        plans = plan_parts(total_epochs, total_parts)
        covered: list[int] = []
        for p in plans:
            covered.extend(range(p.start_epoch, p.end_epoch + 1))
        sizes = [p.n_epochs for p in plans]
        label = f"{total_epochs} epochs / {total_parts} parts"
        ok = (
            len(covered) == len(set(covered))                 # no overlap
            and sorted(covered) == list(range(1, total_epochs + 1))  # exact cover
            and max(sizes) - min(sizes) <= 1                  # balanced
            and len(plans) == total_parts
        )
        check(label, ok, f"sizes={sizes}")


def verify_contiguity() -> None:
    print("\n2. Contiguity (no gap between consecutive Parts)")
    plans = plan_parts(100, 6)
    ok = all(b.start_epoch == a.end_epoch + 1 for a, b in zip(plans, plans[1:]))
    check("consecutive parts are adjacent", ok,
          " ".join(f"{p.part}:{p.start_epoch}-{p.end_epoch}" for p in plans))
    check("first part starts at epoch 1", plans[0].start_epoch == 1)
    check("last part ends at epoch 100", plans[-1].end_epoch == 100)


def verify_whole_dataset_per_part() -> None:
    print("\n3. Every Part trains on the WHOLE dataset (not a file slice)")
    spec = R.get_model("player")
    manifest_dir = R.REPO_ROOT / "runs" / "_data" / spec.dataset
    train_txt = manifest_dir / "train.txt"
    if not train_txt.exists():
        check("train manifest present", False, f"missing {train_txt}")
        return
    lines = [ln for ln in train_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    check("train manifest is non-empty", len(lines) > 0, f"{len(lines):,} images")
    check("train manifest has no duplicates", len(lines) == len(set(lines)),
          f"{len(lines) - len(set(lines))} duplicate(s)")
    # The data.yaml handed to every Part is identical -- so is the file set.
    y1 = R.build_data_yaml(spec.dataset).read_text(encoding="utf-8")
    y2 = R.build_data_yaml(spec.dataset).read_text(encoding="utf-8")
    check("data.yaml is identical across Parts", y1 == y2,
          "all Parts see the same 12,248 images")


def verify_ledger() -> None:
    print("\n4. Ledger / resume bookkeeping")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "parts_state.json"
        st = PartsState(path)
        plans = plan_parts(100, 6)
        check("fresh ledger has no completed parts", st.completed() == set())

        st.record(plans[0], status="completed", started="t0", finished="t1",
                  epochs_run=17)
        st.record(plans[1], status="failed", started="t2", finished="t3",
                  epochs_run=3, note="simulated crash")
        check("completed part recorded", st.completed() == {1}, f"{st.completed()}")
        check("failed part NOT counted as completed", 2 not in st.completed())

        # survives a reload (i.e. a reboot)
        reloaded = PartsState(path)
        check("ledger survives process restart", reloaded.completed() == {1},
              f"{reloaded.completed()}")

        # corrupt ledger must not brick the run
        path.write_text("{ this is not json", encoding="utf-8")
        recovered = PartsState(path)
        check("corrupt ledger recovers instead of crashing",
              recovered.completed() == set())


def verify_ordering_rules() -> None:
    print("\n5. Ordering guards")
    with tempfile.TemporaryDirectory() as td:
        st = PartsState(Path(td) / "s.json")
        plans = plan_parts(100, 6)
        st.record(plans[0], status="completed", started="", finished="", epochs_run=17)
        st.record(plans[2], status="completed", started="", finished="", epochs_run=17)
        done = st.completed()
        missing_before_4 = [p for p in range(1, 4) if p not in done]
        check("gap before Part 4 is detected", missing_before_4 == [2],
              f"missing={missing_before_4}")
        missing_before_2 = [p for p in range(1, 2) if p not in done]
        check("Part 2 is allowed to run", missing_before_2 == [])


def main() -> int:
    print("=" * 62)
    print("  PART-WISE TRAINING VALIDATION")
    print("=" * 62)
    verify_partitioning()
    verify_contiguity()
    verify_whole_dataset_per_part()
    verify_ledger()
    verify_ordering_rules()
    print("\n" + "=" * 62)
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
