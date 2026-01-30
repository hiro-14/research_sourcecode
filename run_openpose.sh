#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/openpose_research.py"   --base-dir "${SCRIPT_DIR}"   --views json_output/view1 json_output/view2 json_output/view3 json_output/view4   --p-mats P_matrix   --out-dir analysis_output   --fps 30   --person-index 0
