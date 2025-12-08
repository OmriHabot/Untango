import argparse
import json
import time
import math
import httpx
import statistics
from typing import List, Dict, Any, Optional
from scipy import stats as scipy_stats
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.markdown import Markdown

console = Console()
API_BASE = "http://localhost:8001"

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        return json.load(f)

def run_rag_query(query: str, client: httpx.Client, config: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    try:
        # Default RAG settings
        payload = {
            "query": query,
            "n_results": 10,
            "confidence_threshold": 0.01  # Low threshold to ensure we get results for eval
        }
        
        # Apply ablation config
        if config.get("disable_bm25"):
            payload["bm25_score_threshold"] = 10000.0 # Impossible score
        if config.get("disable_vector"):
            payload["vector_similarity_threshold"] = 1.1 # Impossible similarity
            
        response = client.post(
            f"{API_BASE}/query-db",
            json=payload,
            timeout=30.0
        )
        
        if response.status_code == 404: 
             return {"result": None, "latency": time.time() - start_time, "success": False, "error": "No results"}
             
        response.raise_for_status()
        result = response.json()
        latency = time.time() - start_time
        return {"result": result, "latency": latency, "success": True}
    except Exception as e:
        return {"result": None, "latency": time.time() - start_time, "success": False, "error": str(e)}

def evaluate_retrieval(retrieved_chunks: List[Dict], expected_keywords: List[str]) -> Dict[str, float]:
    hits = 0
    ranks = []
    found_any = False
    
    for i, chunk in enumerate(retrieved_chunks):
        content = chunk.get("content", "").lower()
        metadata = str(chunk.get("metadata", "")).lower()
        match = any(k.lower() in content or k.lower() in metadata for k in expected_keywords)
        if match:
            if not found_any:
                ranks.append(1 / (i + 1))
                found_any = True
            hits += 1
            
    return {
        "hit_rate": 1.0 if hits > 0 else 0.0,
        "mrr": ranks[0] if ranks else 0.0
    }

def run_evaluation_suite(dataset: List[Dict], suite_name: str, description: str, config: Dict[str, Any]) -> Dict[str, Any]:
    console.print(Panel(Text(suite_name, style="bold magenta"), title="Running Suite", border_style="magenta", expand=False))
    console.print(Markdown(description))
    console.print()

    metrics = {
        "latencies": [],
        "hit_rates": [],
        "mrrs": []
    }
    
    # Error tracking for analysis
    errors = {
        "no_results": [],
        "api_error": [],
        "timeout": []
    }
    
    with httpx.Client() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {len(dataset)} queries...", total=len(dataset))
            
            for item in dataset:
                rag_out = run_rag_query(item['query'], client, config)
                
                if not rag_out["success"]:
                    # Track error type
                    error_msg = rag_out.get("error", "unknown")
                    if "timeout" in error_msg.lower():
                        errors["timeout"].append({"query": item['query'], "error": error_msg})
                    elif "no results" in error_msg.lower():
                        errors["no_results"].append({"query": item['query'], "error": error_msg})
                    else:
                        errors["api_error"].append({"query": item['query'], "error": error_msg})
                    
                    # Penalize failures
                    metrics["hit_rates"].append(0.0)
                    metrics["mrrs"].append(0.0)
                else:
                    metrics["latencies"].append(rag_out["latency"])
                    
                    retrieval_scores = evaluate_retrieval(
                        rag_out["result"].get("retrieved_chunks", []), 
                        item.get("expected_chunks", []) # keywords
                    )
                    metrics["hit_rates"].append(retrieval_scores["hit_rate"])
                    metrics["mrrs"].append(retrieval_scores["mrr"])
                
                progress.advance(task)
    
    # Calculate comprehensive statistics
    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}
        
        n = len(values)
        mean = statistics.mean(values)
        
        if n < 2:
            return {"mean": mean, "std": 0.0, "ci_95_lower": mean, "ci_95_upper": mean}
        
        std = statistics.stdev(values)
        
        # 95% CI using t-distribution
        se = std / math.sqrt(n)
        t_value = scipy_stats.t.ppf(0.975, n - 1)  # 95% CI, two-tailed
        ci_margin = t_value * se
        
        return {
            "mean": mean,
            "std": std,
            "ci_95_lower": mean - ci_margin,
            "ci_95_upper": mean + ci_margin
        }
    
    latency_stats = calc_stats(metrics["latencies"])
    hit_rate_stats = calc_stats(metrics["hit_rates"])
    mrr_stats = calc_stats(metrics["mrrs"])
    
    stats = {
        # Legacy fields for backwards compatibility
        "avg_latency": latency_stats["mean"],
        "avg_hit_rate": hit_rate_stats["mean"],
        "avg_mrr": mrr_stats["mean"],
        # New statistical fields
        "std_latency": latency_stats["std"],
        "std_hit_rate": hit_rate_stats["std"],
        "std_mrr": mrr_stats["std"],
        "ci_95_hit_rate": (hit_rate_stats["ci_95_lower"], hit_rate_stats["ci_95_upper"]),
        "ci_95_mrr": (mrr_stats["ci_95_lower"], mrr_stats["ci_95_upper"]),
        "ci_95_latency": (latency_stats["ci_95_lower"], latency_stats["ci_95_upper"]),
        # Sample size
        "n_queries": len(dataset),
        "n_successful": len(metrics["latencies"]),
        # Error summary
        "errors": {
            "no_results": len(errors["no_results"]),
            "api_error": len(errors["api_error"]),
            "timeout": len(errors["timeout"]),
            "details": errors
        }
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline with Ablation")
    parser.add_argument("--dataset", default="tests/evaluation_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study (Hybrid vs Vector vs BM25)")
    parser.add_argument("--output-file", default=None, help="Path to save results as JSON")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]Untango RAG Evaluation Suite[/bold green]\n"
        "This tool evaluates the retrieval performance of the RAG pipeline.\n"
        "It measures Hit Rate, Mean Reciprocal Rank (MRR), and Latency.",
        title="Welcome",
        border_style="green"
    ))

    try:
        dataset = load_dataset(args.dataset)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Dataset not found at {args.dataset}")
        return

    if args.limit:
        dataset = dataset[:args.limit]

    # Description of metrics
    metrics_table = Table(title="Evaluation Metrics Explained", box=box.SIMPLE)
    metrics_table.add_column("Metric", style="cyan", no_wrap=True)
    metrics_table.add_column("Description")
    metrics_table.add_row("Hit Rate", "Percentage of queries where at least one relevant chunk was retrieved.")
    metrics_table.add_row("MRR (Mean Reciprocal Rank)", "Measures how high up the relevant chunk appears. 1.0 means top result is relevant.")
    metrics_table.add_row("Latency", "Time taken to process the query and retrieve results.")
    console.print(metrics_table)
    console.print()

    results_data = {
        "timestamp": time.time(),
        "dataset_size": len(dataset),
        "metrics": {}
    }

    # 1. Hybrid (Baseline)
    hybrid_desc = (
        "**Hybrid Search** combines keyword matching (BM25) with semantic vector search.\n"
        "This is the default configuration for the production application."
    )
    hybrid_stats = run_evaluation_suite(dataset, "Hybrid Search (Default)", hybrid_desc, {})
    results_data["metrics"]["hybrid"] = hybrid_stats
    
    if args.ablation:
        # 2. Vector Only
        vector_desc = (
            "**Vector Only Search** relies solely on semantic embeddings.\n"
            "Good for capturing meaning but might miss exact keyword matches."
        )
        vector_stats = run_evaluation_suite(dataset, "Vector Only", vector_desc, {"disable_bm25": True})
        results_data["metrics"]["vector"] = vector_stats
        
        # 3. BM25 Only
        bm25_desc = (
            "**BM25 (Keyword) Only** relies solely on exact keyword matching.\n"
            "Good for specific terms but fails on synonyms or semantic meaning."
        )
        bm25_stats = run_evaluation_suite(dataset, "BM25 Only", bm25_desc, {"disable_vector": True})
        results_data["metrics"]["bm25"] = bm25_stats
        
        console.print()
        results_table = Table(title="Comparative Analysis (Ablation Results)", box=box.ROUNDED)
        results_table.add_column("Mode", style="magenta")
        results_table.add_column("Hit Rate", justify="right", style="green")
        results_table.add_column("MRR", justify="right", style="blue")
        results_table.add_column("Latency", justify="right", style="yellow")
        
        results_table.add_row(
            "Hybrid", 
            f"{hybrid_stats['avg_hit_rate']:.2%}", 
            f"{hybrid_stats['avg_mrr']:.4f}", 
            f"{hybrid_stats['avg_latency']:.2f}s"
        )
        results_table.add_row(
            "Vector Only", 
            f"{vector_stats['avg_hit_rate']:.2%}", 
            f"{vector_stats['avg_mrr']:.4f}", 
            f"{vector_stats['avg_latency']:.2f}s"
        )
        results_table.add_row(
            "BM25 Only", 
            f"{bm25_stats['avg_hit_rate']:.2%}", 
            f"{bm25_stats['avg_mrr']:.4f}", 
            f"{bm25_stats['avg_latency']:.2f}s"
        )
        
        console.print(results_table)
    else:
        # Single run results
        results_table = Table(title="Evaluation Results", box=box.ROUNDED)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", justify="right", style="green")
        
        results_table.add_row("Hit Rate", f"{hybrid_stats['avg_hit_rate']:.2%}")
        results_table.add_row("MRR", f"{hybrid_stats['avg_mrr']:.4f}")
        results_table.add_row("Avg Latency", f"{hybrid_stats['avg_latency']:.2f}s")
        
        console.print(results_table)

    if args.output_file:
        try:
            # Convert tuples to lists for JSON serialization
            def serialize_stats(stats_dict):
                serialized = {}
                for k, v in stats_dict.items():
                    if isinstance(v, tuple):
                        serialized[k] = list(v)
                    elif isinstance(v, dict):
                        serialized[k] = serialize_stats(v)
                    else:
                        serialized[k] = v
                return serialized
            
            serialized_data = {
                "timestamp": results_data["timestamp"],
                "dataset_size": results_data["dataset_size"],
                "metrics": {k: serialize_stats(v) for k, v in results_data["metrics"].items()}
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            console.print(f"[bold green]Results saved to {args.output_file}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error saving results: {e}[/bold red]")
    
    # Print statistical significance section if ablation was run
    if args.ablation:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Statistical Significance Analysis[/bold cyan]",
            title="📊 Statistics",
            border_style="cyan"
        ))
        
        # Detailed statistics table
        stats_table = Table(title="Detailed Statistics (Mean ± Std, 95% CI)", box=box.ROUNDED)
        stats_table.add_column("Configuration", style="magenta")
        stats_table.add_column("Hit Rate (Mean ± Std)", justify="center", style="green")
        stats_table.add_column("Hit Rate 95% CI", justify="center", style="green")
        stats_table.add_column("MRR (Mean ± Std)", justify="center", style="blue")
        stats_table.add_column("MRR 95% CI", justify="center", style="blue")
        
        for name, s in [("Hybrid", hybrid_stats), ("Vector Only", vector_stats), ("BM25 Only", bm25_stats)]:
            hr_ci = s.get("ci_95_hit_rate", (0, 0))
            mrr_ci = s.get("ci_95_mrr", (0, 0))
            stats_table.add_row(
                name,
                f"{s['avg_hit_rate']:.2%} ± {s.get('std_hit_rate', 0):.2%}",
                f"[{hr_ci[0]:.2%}, {hr_ci[1]:.2%}]",
                f"{s['avg_mrr']:.4f} ± {s.get('std_mrr', 0):.4f}",
                f"[{mrr_ci[0]:.4f}, {mrr_ci[1]:.4f}]"
            )
        
        console.print(stats_table)
        console.print()
        
        # Error Analysis Summary
        error_table = Table(title="Error Analysis Summary", box=box.SIMPLE)
        error_table.add_column("Configuration", style="magenta")
        error_table.add_column("Total Queries", justify="right")
        error_table.add_column("Successful", justify="right", style="green")
        error_table.add_column("No Results", justify="right", style="yellow")
        error_table.add_column("API Errors", justify="right", style="red")
        error_table.add_column("Timeouts", justify="right", style="red")
        
        for name, s in [("Hybrid", hybrid_stats), ("Vector Only", vector_stats), ("BM25 Only", bm25_stats)]:
            errs = s.get("errors", {})
            error_table.add_row(
                name,
                str(s.get("n_queries", 0)),
                str(s.get("n_successful", 0)),
                str(errs.get("no_results", 0)),
                str(errs.get("api_error", 0)),
                str(errs.get("timeout", 0))
            )
        
        console.print(error_table)

if __name__ == "__main__":
    main()
