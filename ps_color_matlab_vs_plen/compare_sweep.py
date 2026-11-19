"""Compare each MATLAB reference in sweep.json with fresh plenoptic statistics."""

import itertools
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from color_test_helpers import load_matlab_statistics
from compare_statistics import compare_color_ps_statistics, write_results

torch.set_num_threads(2)
atol = 1e-4
output_dir = Path("validation")
sweep = json.loads(Path("sweep.json").read_text())
cases = []
for image, scales, orientations, width in itertools.product(
    sweep["images"],
    sweep["n_scales"],
    sweep["n_orientations"],
    sweep["spatial_corr_width"],
):
    case_id = f"{image}-scales{scales}-ori{orientations}-spat{width}"
    cases.append(
        {
            "case_id": case_id,
            "image_name": image,
            "image_path": f"../inputs/{image}.tif",
            "n_scales": scales,
            "n_orientations": orientations,
            "spatial_corr_width": width,
            "matlab_path": f"matlab_statistics/{case_id}.mat",
        }
    )
assert cases, "Empty parameter sweep"
assert len({case["case_id"] for case in cases}) == len(cases), "Duplicate sweep cases"

missing = [
    case["matlab_path"]
    for case in cases
    if not (output_dir / case["matlab_path"]).is_file()
]
assert not missing, (
    f"Missing {len(missing)} MATLAB references; generate them on the cluster "
    f"with export_matlab_ps_stats and copy them back. First missing: {missing[0]}"
)

comparisons = []
summaries = []
for case in cases:
    matlab = load_matlab_statistics(output_dir / case["matlab_path"])
    assert (matlab["Nsc"], matlab["Nor"], matlab["Na"]) == (
        case["n_scales"],
        case["n_orientations"],
        case["spatial_corr_width"],
    ), f"{case['case_id']}: cached parameters differ; regenerate this reference"
    assert np.array_equal(matlab["im0"], iio.imread(output_dir / case["image_path"])), (
        f"{case['case_id']}: cached image differs; regenerate this reference"
    )
    rows = compare_color_ps_statistics(matlab, atol=atol, verbose=False)
    comparisons.extend({**case, **row} for row in rows)
    failures = [row for row in rows if row["status"] == "failed"]
    summary = {
        **case,
        "n_statistic_families": len(rows),
        "n_failed_families": len(failures),
        "atol": atol,
        "status": "failed" if failures else "passed",
    }
    summaries.append(summary)
    print(f"{case['case_id']}: {summary['status']}", flush=True)
    for row in failures:
        print(
            f"  {row['plenoptic_name']}: "
            f"max absolute error {row['max_absolute_difference']:.6g}",
            flush=True,
        )

write_results(comparisons, output_dir / "comparisons.csv")
write_results(summaries, output_dir / "case_summary.csv")
n_failed = sum(row["status"] == "failed" for row in summaries)
print(f"{len(cases) - n_failed}/{len(cases)} cases passed; results in {output_dir}")
raise SystemExit(int(n_failed > 0))
