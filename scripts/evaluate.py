import argparse
import json
import time
import httpx
import statistics
from typing import List, Dict, Any
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

def run_evaluation_suite(dataset: List[Dict], suite_name: str, description: str, config: Dict[str, Any]) -> Dict[str, float]:
    console.print(Panel(Text(suite_name, style="bold magenta"), title="Running Suite", border_style="magenta", expand=False))
    console.print(Markdown(description))
    console.print()

    metrics = {
        "latencies": [],
        "hit_rates": [],
        "mrrs": []
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
            
    stats = {
        "avg_latency": statistics.mean(metrics["latencies"]) if metrics["latencies"] else 0.0,
        "avg_hit_rate": statistics.mean(metrics["hit_rates"]) if metrics["hit_rates"] else 0.0,
        "avg_mrr": statistics.mean(metrics["mrrs"]) if metrics["mrrs"] else 0.0
    }
    
    # console.print(f"  Avg Hit Rate: {stats['avg_hit_rate']:.2%}")
    # console.print(f"  Avg MRR:      {stats['avg_mrr']:.4f}")
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
            with open(args.output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            console.print(f"[bold green]Results saved to {args.output_file}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error saving results: {e}[/bold red]")

if __name__ == "__main__":
    main()
