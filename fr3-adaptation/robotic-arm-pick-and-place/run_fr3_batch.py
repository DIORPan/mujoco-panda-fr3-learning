import json
import subprocess
import sys
import time
from pathlib import Path

batch_dir = Path("runs/fr3_migration") / f"batch_{time.time_ns()}"
batch_dir.mkdir(parents=True)
summary = {"requested_seeds": list(range(10)), "rounds": []}

print("批次目录：", batch_dir.resolve(), flush=True)

for seed in summary["requested_seeds"]:
    seed_dir = batch_dir / f"seed_{seed}"
    seed_dir.mkdir()

    with (seed_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable, "-X", "utf8", "-B",
                "run_fr3_random.py",
                "--seed", str(seed),
                "--headless",
                "--output-dir", str(seed_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    report_path = next(seed_dir.glob("random_seed*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    passed = sum(row["success"] for row in report["results"])
    failed = [
        row["color"] for row in report["results"]
        if not row["success"]
    ]

    summary["rounds"].append({
        "seed": seed,
        "completed": report["completed"],
        "passed": passed,
        "failed_colors": failed,
        "report": str(report_path.relative_to(batch_dir)),
    })
    (batch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"seed={seed}: {passed}/3，失败方块={failed}", flush=True)

passed_cubes = sum(row["passed"] for row in summary["rounds"])
passed_rounds = sum(
    row["completed"] and row["passed"] == 3
    for row in summary["rounds"]
)
print(f"方块抓放通过：{passed_cubes}/30")
print(f"三块全部通过的回合：{passed_rounds}/10")
print("汇总文件：", (batch_dir / "summary.json").resolve())