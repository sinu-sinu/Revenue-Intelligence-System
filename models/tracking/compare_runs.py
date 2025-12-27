"""
CLI Tool for Comparing Experiment Runs

Usage:
    python -m models.tracking.compare_runs
    python -m models.tracking.compare_runs --top 5
    python -m models.tracking.compare_runs --metric brier_score --ascending
"""

import argparse
from typing import List, Dict, Any
from models.tracking.experiment_tracker import ExperimentTracker


def format_run_table(runs: List[Dict[str, Any]], metrics: List[str]) -> str:
    """
    Format runs as a table.
    
    Args:
        runs: List of run dictionaries
        metrics: List of metric names to display
        
    Returns:
        Formatted table string
    """
    if not runs:
        return "No runs found."
    
    # Build header
    header = ["Run ID", "Date", "Status"]
    header.extend(metrics)
    header.append("Key Params")
    
    # Build rows
    rows = []
    for run in runs:
        run_id = run.get("run_id", "unknown")[:20]
        date = run.get("start_time", "")[:10]
        status = run.get("status", "unknown")[:8]
        
        # Get metrics
        run_metrics = run.get("metrics", {})
        metric_values = [f"{run_metrics.get(m, 'N/A'):.4f}" if isinstance(run_metrics.get(m), (int, float)) else "N/A" 
                        for m in metrics]
        
        # Get key params
        params = run.get("params", {})
        key_params = []
        if "num_leaves" in params:
            key_params.append(f"leaves={params['num_leaves']}")
        if "lambda_l2" in params:
            key_params.append(f"l2={params['lambda_l2']}")
        if "min_data_in_leaf" in params:
            key_params.append(f"min_leaf={params['min_data_in_leaf']}")
        param_str = ", ".join(key_params) if key_params else "default"
        
        row = [run_id, date, status] + metric_values + [param_str]
        rows.append(row)
    
    # Calculate column widths
    all_rows = [header] + rows
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(header))]
    
    # Format output
    lines = []
    
    # Header
    header_line = " | ".join(str(header[i]).ljust(widths[i]) for i in range(len(header)))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in rows:
        row_line = " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row)))
        lines.append(row_line)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of runs to show (default: 10)"
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default="auc_roc",
        help="Metric to sort by (default: auc_roc)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Filter by experiment name"
    )
    parser.add_argument(
        "--tracking-dir",
        type=str,
        default="experiments",
        help="Tracking directory (default: experiments)"
    )
    
    args = parser.parse_args()
    
    # Load runs
    runs = ExperimentTracker.load_all_runs(args.tracking_dir)
    
    if not runs:
        print("No experiment runs found.")
        print(f"Run training with: python train.py")
        return
    
    # Filter by experiment if specified
    if args.experiment:
        runs = [r for r in runs if r.get("experiment_name") == args.experiment]
    
    # Sort by metric
    def get_metric(run):
        return run.get("metrics", {}).get(args.metric, float("-inf") if not args.ascending else float("inf"))
    
    runs.sort(key=get_metric, reverse=not args.ascending)
    
    # Limit to top N
    runs = runs[:args.top]
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT RUNS (sorted by {args.metric}, {'asc' if args.ascending else 'desc'})")
    print(f"{'='*60}\n")
    
    # Format and print table
    metrics = ["auc_roc", "brier_score", "ece"]
    table = format_run_table(runs, metrics)
    print(table)
    
    # Print best run
    best = ExperimentTracker.get_best_run(args.tracking_dir, args.metric, not args.ascending)
    if best:
        print(f"\nBest run by {args.metric}: {best['run_id']}")
        print(f"  AUC: {best['metrics'].get('auc_roc', 'N/A')}")
        print(f"  Brier: {best['metrics'].get('brier_score', 'N/A')}")
    
    print()


if __name__ == "__main__":
    main()

