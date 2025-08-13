import json
import sys

def main(path: str) -> None:
    """Simple grading script for GitHub Classroom.

    Reads a metrics JSON file produced by the evaluation script and prints a grade line
    based on macro‑F1 thresholds.
    """
    with open(path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    f1 = float(metrics.get('f1_macro', 0.0))

    if f1 >= 0.65:
        pts, band = 45, '≥10% above baseline'
    elif f1 >= 0.60:
        pts, band = 40, '≈10% above baseline'
    elif f1 >= 0.55:
        pts, band = 35, '≥5% above baseline'
    else:
        pts, band = 25, 'below baseline'
    print(f"GRADE: model_quality_points={pts} (macro-F1={f1:.3f}, {band})")

if __name__ == '__main__':
    # Default to reports/metrics.json if no argument provided
    main(sys.argv[1] if len(sys.argv) > 1 else 'reports/metrics.json')
