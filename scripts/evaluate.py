"""
RAG Evaluation Suite with Hugging Face Dataset Support

This script evaluates the RAG pipeline by:
1. Downloading a standard QA dataset from HuggingFace (e.g., SQuAD)
2. Ingesting ALL context passages from the dataset into ChromaDB
3. Querying with the questions from the dataset  
4. Checking if the ground truth answer appears in the retrieved content

Metrics calculated:
- Hit Rate: Percentage of queries where the answer text is found in retrieved chunks
- MRR (Mean Reciprocal Rank): Measures how high the answer appears in results
- Context Relevance: Jaccard similarity between retrieved and reference context
- Latency: Query processing time
"""

import argparse
import json
import time
import math
import httpx
import statistics
import hashlib
import sys
import os
from typing import List, Dict, Any, Optional, Tuple
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

# Available Hugging Face datasets for RAG evaluation
# Each dataset has different column names for question, answer, and context
HF_DATASETS = {
    "squad": {
        "name": "rajpurkar/squad",
        "description": "SQuAD: Stanford Question Answering Dataset (Wikipedia passages)",
        "question_col": "question",
        "answer_col": "answers",  # Dict with {"text": [...], "answer_start": [...]}
        "context_col": "context",
        "split": "validation",
        "answer_extractor": lambda x: x.get("text", []) if isinstance(x, dict) else [x]
    },
    "squad_train": {
        "name": "rajpurkar/squad",
        "description": "SQuAD Training Set (larger, 87k samples)",
        "question_col": "question",
        "answer_col": "answers",
        "context_col": "context",
        "split": "train",
        "answer_extractor": lambda x: x.get("text", []) if isinstance(x, dict) else [x]
    },
    "wiki_qa": {
        "name": "microsoft/wiki_qa",
        "description": "WikiQA: Open-domain QA from Bing queries + Wikipedia",
        "question_col": "question",
        "answer_col": "answer",  # The sentence column when label=1
        "context_col": "answer",  # Same as answer for wiki_qa (sentence pairs)
        "split": "test",
        "answer_extractor": lambda x: [x] if isinstance(x, str) else x,
        "filter_fn": lambda item: item.get("label", 0) == 1  # Only positive examples
    },
    "trivia_qa": {
        "name": "trivia_qa",
        "description": "TriviaQA: Reading comprehension with trivia questions",
        "question_col": "question",
        "answer_col": "answer",  # Dict with {"value": "...", "aliases": [...]}
        "context_col": "search_results",  # Contains search context
        "split": "validation",
        "subset": "rc.nocontext",  # Use nocontext to avoid huge context fields
        "answer_extractor": lambda x: [x.get("value", "")] + x.get("aliases", []) if isinstance(x, dict) else [x]
    }
}

DEFAULT_HF_DATASET = "squad"


def load_local_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from local JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_hf_dataset(dataset_key: str, limit: int = 200) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]:
    """
    Load a RAG evaluation dataset from Hugging Face.
    
    Returns:
        (samples, contexts, dataset_info)
        - samples: List of {id, query, context, context_id, answer_texts}
        - contexts: Dict mapping context_id -> context text
        - dataset_info: Metadata about the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[bold red]Error:[/bold red] 'datasets' library not installed. Run: pip install datasets")
        raise

    if dataset_key not in HF_DATASETS:
        console.print(f"[bold red]Error:[/bold red] Unknown dataset '{dataset_key}'. Available: {list(HF_DATASETS.keys())}")
        raise ValueError(f"Unknown dataset: {dataset_key}")
    
    config = HF_DATASETS[dataset_key]
    console.print(f"[cyan]Loading dataset:[/cyan] {config['name']} ({config['description']})")
    
    try:
        # Load dataset with optional subset
        if "subset" in config:
            dataset = load_dataset(config["name"], config["subset"], split=config["split"])
        else:
            dataset = load_dataset(config["name"], split=config["split"])
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise
    
    # Convert to evaluation format
    samples = []
    unique_contexts = {}  # Deduplicate contexts by hash
    answer_extractor = config.get("answer_extractor", lambda x: [x] if isinstance(x, str) else x)
    filter_fn = config.get("filter_fn", lambda x: True)
    
    for i, item in enumerate(dataset):
        if len(samples) >= limit:
            break
        
        # Apply filter if defined (e.g., wiki_qa only uses label=1)
        if not filter_fn(item):
            continue
        
        question = item.get(config["question_col"], "")
        context_raw = item.get(config["context_col"], "")
        answers_field = item.get(config["answer_col"], {})
        
        # Handle context that might be a list or nested structure
        if isinstance(context_raw, list):
            context = " ".join(str(c) for c in context_raw[:3])  # Take first 3 if list
        elif isinstance(context_raw, dict):
            context = str(context_raw.get("text", context_raw.get("wiki_context", "")))
        else:
            context = str(context_raw) if context_raw else ""
        
        # Extract answer texts using the dataset-specific extractor
        answer_texts = answer_extractor(answers_field)
        if not isinstance(answer_texts, list):
            answer_texts = [answer_texts]
        answer_texts = [str(a) for a in answer_texts if a]
        
        if not question or not answer_texts:
            continue
        
        # Create unique ID for context (or use empty if no context)
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8] if context else "no_ctx"
        if context:
            unique_contexts[context_hash] = context
        
        sample = {
            "id": f"hf_{len(samples)}",
            "query": question,
            "context": context,
            "context_id": context_hash,
            "answer_texts": answer_texts
        }
        samples.append(sample)
    
    dataset_info = {
        "name": config["name"],
        "description": config["description"],
        "total_samples": len(dataset),
        "samples_used": len(samples),
        "unique_contexts": len(unique_contexts),
        "split": config["split"]
    }
    
    console.print(f"[green]✓ Loaded {len(samples)} samples with {len(unique_contexts)} unique contexts[/green]")
    return samples, unique_contexts, dataset_info


def ingest_contexts_to_chromadb(contexts: Dict[str, str]) -> bool:
    """
    Ingest context passages directly into ChromaDB.
    Uses the same collection and embedding function as the main app.
    """
    console.print(Panel.fit(
        f"[bold cyan]Ingesting {len(contexts)} Dataset Contexts[/bold cyan]\n"
        "Uploading passages to ChromaDB with embeddings...",
        title="📥 Ingestion",
        border_style="cyan"
    ))
    
    try:
        # Import ChromaDB utilities from the app
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.database import get_collection

        
        collection = get_collection()
        
        # Prepare batch data
        ids = []
        documents = []
        metadatas = []
        
        for ctx_id, ctx_text in contexts.items():
            ids.append(f"hf_context_{ctx_id}")
            documents.append(ctx_text)
            metadatas.append({
                "source": "huggingface_eval",
                "context_id": ctx_id,
                "chunk_type": "evaluation_context"
            })
        
        # Add to collection in batches (ChromaDB has batch limits)
        batch_size = 100
        total_added = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Ingesting contexts...", total=len(ids))
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                
                # Use upsert to handle duplicates
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                total_added += len(batch_ids)
                progress.update(task, advance=len(batch_ids))
        
        console.print(f"[green]✓ Ingested {total_added} context passages[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        console.print("[yellow]Falling back to API-based query evaluation...[/yellow]")
        return False


def run_rag_query(query: str, client: httpx.Client, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a RAG query and return results."""
    start_time = time.time()
    try:
        payload = {
            "query": query,
            "n_results": 10,
            "confidence_threshold": 0.001  # Very low to get results
        }
        
        # Apply ablation config
        if config.get("disable_bm25"):
            payload["bm25_score_threshold"] = 10000.0
        if config.get("disable_vector"):
            payload["vector_similarity_threshold"] = 1.1
            
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
    except httpx.TimeoutException:
        return {"result": None, "latency": time.time() - start_time, "success": False, "error": "Timeout"}
    except Exception as e:
        return {"result": None, "latency": time.time() - start_time, "success": False, "error": str(e)}


def evaluate_retrieval(
    retrieved_chunks: List[Dict], 
    answer_texts: List[str],
    reference_context: str = ""
) -> Dict[str, float]:
    """
    Evaluate retrieval quality by checking if answer appears in retrieved content.
    
    The hit is counted if ANY of the answer_texts (lowercased) appears as a substring
    in ANY of the retrieved chunks (lowercased).
    
    Args:
        retrieved_chunks: List of retrieved chunks with 'content' field
        answer_texts: List of valid answer strings (any match counts as hit)
        reference_context: Original context passage for relevance calculation
    
    Returns:
        hit_rate: 1.0 if any answer text found in any chunk, 0.0 otherwise
        mrr: Reciprocal rank of first chunk containing answer
        context_relevance: Jaccard similarity with reference
    """
    first_hit_rank = None
    all_retrieved_text = ""
    
    for i, chunk in enumerate(retrieved_chunks):
        content = chunk.get("content", "").lower()
        all_retrieved_text += content + " "
        
        # Check if any answer appears in this chunk (case-insensitive substring match)
        for ans in answer_texts:
            ans_lower = ans.lower().strip()
            if ans_lower and ans_lower in content:
                if first_hit_rank is None:
                    first_hit_rank = i + 1  # 1-indexed rank
                break
    
    # Calculate hit rate (1 if answer found anywhere)
    hit = 1.0 if first_hit_rank is not None else 0.0
    
    # Calculate MRR
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
    
    # Calculate context relevance (Jaccard similarity)
    context_relevance = 0.0
    if reference_context and all_retrieved_text:
        ref_words = set(reference_context.lower().split())
        ret_words = set(all_retrieved_text.split())
        if ref_words or ret_words:
            intersection = len(ref_words & ret_words)
            union = len(ref_words | ret_words)
            context_relevance = intersection / union if union > 0 else 0.0
            
    return {
        "hit_rate": hit,
        "mrr": mrr,
        "context_relevance": min(context_relevance, 1.0)
    }


def run_evaluation_suite(
    samples: List[Dict], 
    suite_name: str, 
    description: str, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run evaluation suite and collect metrics."""
    console.print(Panel(Text(suite_name, style="bold magenta"), title="Running Suite", border_style="magenta", expand=False))
    console.print(Markdown(description))
    console.print()

    metrics = {
        "latencies": [],
        "hit_rates": [],
        "mrrs": [],
        "context_relevances": []
    }
    
    errors = {
        "no_results": [],
        "api_error": [],
        "timeout": []
    }
    
    query_results = []
    
    with httpx.Client() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {len(samples)} queries...", total=len(samples))
            
            for item in samples:
                rag_out = run_rag_query(item['query'], client, config)
                
                query_result = {
                    "query": item['query'][:100],
                    "answer_texts": item.get("answer_texts", [])[:3],
                    "success": rag_out["success"],
                    "latency": rag_out["latency"]
                }
                
                if not rag_out["success"]:
                    error_msg = rag_out.get("error", "unknown")
                    if "timeout" in error_msg.lower():
                        errors["timeout"].append({"query": item['query'][:50], "error": error_msg})
                    elif "no results" in error_msg.lower():
                        errors["no_results"].append({"query": item['query'][:50], "error": error_msg})
                    else:
                        errors["api_error"].append({"query": item['query'][:50], "error": error_msg})
                    
                    metrics["hit_rates"].append(0.0)
                    metrics["mrrs"].append(0.0)
                    metrics["context_relevances"].append(0.0)
                    query_result["metrics"] = {"hit_rate": 0.0, "mrr": 0.0, "context_relevance": 0.0}
                else:
                    metrics["latencies"].append(rag_out["latency"])
                    
                    retrieval_scores = evaluate_retrieval(
                        rag_out["result"].get("retrieved_chunks", []), 
                        item.get("answer_texts", []),
                        item.get("context", "")
                    )
                    metrics["hit_rates"].append(retrieval_scores["hit_rate"])
                    metrics["mrrs"].append(retrieval_scores["mrr"])
                    metrics["context_relevances"].append(retrieval_scores["context_relevance"])
                    
                    query_result["metrics"] = retrieval_scores
                
                query_results.append(query_result)
                progress.advance(task)
    
    def calc_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}
        
        n = len(values)
        mean = statistics.mean(values)
        
        if n < 2:
            return {"mean": mean, "std": 0.0, "ci_95_lower": mean, "ci_95_upper": mean}
        
        std = statistics.stdev(values)
        se = std / math.sqrt(n)
        t_value = scipy_stats.t.ppf(0.975, n - 1)
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
    context_rel_stats = calc_stats(metrics["context_relevances"])
    
    stats = {
        "avg_latency": latency_stats["mean"],
        "avg_hit_rate": hit_rate_stats["mean"],
        "avg_mrr": mrr_stats["mean"],
        "avg_context_relevance": context_rel_stats["mean"],
        "std_latency": latency_stats["std"],
        "std_hit_rate": hit_rate_stats["std"],
        "std_mrr": mrr_stats["std"],
        "std_context_relevance": context_rel_stats["std"],
        "ci_95_hit_rate": (hit_rate_stats["ci_95_lower"], hit_rate_stats["ci_95_upper"]),
        "ci_95_mrr": (mrr_stats["ci_95_lower"], mrr_stats["ci_95_upper"]),
        "ci_95_latency": (latency_stats["ci_95_lower"], latency_stats["ci_95_upper"]),
        "n_queries": len(samples),
        "n_successful": len(metrics["latencies"]),
        "errors": {
            "no_results": len(errors["no_results"]),
            "api_error": len(errors["api_error"]),
            "timeout": len(errors["timeout"])
        },
        "query_results": query_results[:10]
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline with HuggingFace Datasets")
    parser.add_argument("--dataset", default="tests/evaluation_dataset.json", help="Path to local evaluation dataset")
    parser.add_argument("--hf-dataset", type=str, default=None, 
                        help=f"Use HuggingFace dataset. Options: {list(HF_DATASETS.keys())}")
    parser.add_argument("--limit", type=int, default=200, help="Limit number of queries")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study (Hybrid vs Vector vs BM25)")
    parser.add_argument("--output-file", default=None, help="Path to save results as JSON")
    parser.add_argument("--json", action="store_true", help="Output results as JSON for frontend consumption")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip context ingestion (use existing data)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold green]Untango RAG Evaluation Suite[/bold green]\n"
        "Evaluate RAG pipeline using HuggingFace datasets.\n"
        "Process: Download dataset → Ingest contexts → Query → Measure metrics",
        title="Welcome",
        border_style="green"
    ))

    # Load dataset
    dataset_info = None
    contexts = {}
    
    if args.hf_dataset:
        samples, contexts, dataset_info = load_hf_dataset(args.hf_dataset, args.limit)
    else:
        try:
            raw_dataset = load_local_dataset(args.dataset)
            samples = []
            for item in raw_dataset[:args.limit]:
                samples.append({
                    "id": item.get("id", ""),
                    "query": item.get("query", ""),
                    "context": "",
                    "answer_texts": item.get("expected_chunks", []) + item.get("keywords", [])
                })
            dataset_info = {
                "name": "Local Dataset",
                "description": f"Custom evaluation dataset from {args.dataset}",
                "samples_used": len(samples),
                "total_samples": len(raw_dataset),
                "unique_contexts": 0
            }
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Dataset not found at {args.dataset}")
            console.print("[yellow]Tip:[/yellow] Use --hf-dataset squad to use a HuggingFace dataset")
            return

    # Display metrics explanation
    metrics_table = Table(title="Evaluation Metrics", box=box.SIMPLE)
    metrics_table.add_column("Metric", style="cyan", no_wrap=True)
    metrics_table.add_column("Description")
    metrics_table.add_row("Hit Rate", "% of queries where answer.lower() is substring of retrieved content.lower()")
    metrics_table.add_row("MRR", "Mean Reciprocal Rank - position of first chunk containing the answer")
    metrics_table.add_row("Context Relevance", "Jaccard word overlap between retrieved and reference context")
    metrics_table.add_row("Latency", "Average query processing time")
    console.print(metrics_table)
    console.print()

    # Display dataset info
    if dataset_info:
        console.print(Panel.fit(
            f"[bold cyan]Dataset: {dataset_info['name']}[/bold cyan]\n"
            f"{dataset_info['description']}\n"
            f"Samples: {dataset_info['samples_used']} / {dataset_info.get('total_samples', 'N/A')}\n"
            f"Unique Contexts: {dataset_info.get('unique_contexts', 'N/A')}",
            title="📊 Dataset Info",
            border_style="cyan"
        ))
        
        # Show sample queries
        sample_table = Table(title="Sample Queries", box=box.ROUNDED)
        sample_table.add_column("ID", style="magenta")
        sample_table.add_column("Question", max_width=50)
        sample_table.add_column("Answer", max_width=30)
        
        for sample in samples[:5]:
            answers = sample.get("answer_texts", [])
            ans_str = answers[0][:30] if answers else "N/A"
            sample_table.add_row(
                sample.get("id", "?"),
                sample.get("query", "")[:50],
                ans_str + ("..." if len(answers) > 0 and len(answers[0]) > 30 else "")
            )
        console.print(sample_table)
        console.print()

    # Ingest contexts if we have them and not skipping
    if contexts and not args.skip_ingest:
        success = ingest_contexts_to_chromadb(contexts)
        if not success:
            console.print("[yellow]Warning: Context ingestion failed. Continuing anyway...[/yellow]")
        
        # Wait a moment for indexing
        console.print("[dim]Waiting for indexing...[/dim]")
        time.sleep(2)
    elif not contexts:
        console.print("[yellow]No contexts to ingest (using existing DB data)[/yellow]")

    results_data = {
        "timestamp": time.time(),
        "dataset_info": dataset_info,
        "dataset_size": len(samples),
        "metrics": {}
    }

    # Run Hybrid (baseline)
    hybrid_desc = (
        "**Hybrid Search** combines BM25 keyword matching with semantic vector search.\n"
        "Uses Reciprocal Rank Fusion (RRF) to merge results."
    )
    hybrid_stats = run_evaluation_suite(samples, "Hybrid Search (Default)", hybrid_desc, {})
    results_data["metrics"]["hybrid"] = hybrid_stats
    
    if args.ablation:
        vector_desc = "**Vector Only** - Pure semantic embedding search."
        vector_stats = run_evaluation_suite(samples, "Vector Only", vector_desc, {"disable_bm25": True})
        results_data["metrics"]["vector"] = vector_stats
        
        bm25_desc = "**BM25 Only** - Traditional keyword matching."
        bm25_stats = run_evaluation_suite(samples, "BM25 Only", bm25_desc, {"disable_vector": True})
        results_data["metrics"]["bm25"] = bm25_stats
        
        # Comparative results table
        console.print()
        results_table = Table(title="Comparative Analysis (Ablation)", box=box.ROUNDED)
        results_table.add_column("Mode", style="magenta")
        results_table.add_column("Hit Rate", justify="right", style="green")
        results_table.add_column("MRR", justify="right", style="blue")
        results_table.add_column("Context Rel.", justify="right", style="cyan")
        results_table.add_column("Latency", justify="right", style="yellow")
        
        for name, stats in [("Hybrid", hybrid_stats), ("Vector Only", vector_stats), ("BM25 Only", bm25_stats)]:
            results_table.add_row(
                name, 
                f"{stats['avg_hit_rate']:.2%}", 
                f"{stats['avg_mrr']:.4f}", 
                f"{stats['avg_context_relevance']:.2%}",
                f"{stats['avg_latency']:.2f}s"
            )
        console.print(results_table)
    else:
        # Single run results
        results_table = Table(title="Evaluation Results", box=box.ROUNDED)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", justify="right", style="green")
        
        results_table.add_row("Hit Rate", f"{hybrid_stats['avg_hit_rate']:.2%}")
        results_table.add_row("MRR", f"{hybrid_stats['avg_mrr']:.4f}")
        results_table.add_row("Context Relevance", f"{hybrid_stats['avg_context_relevance']:.2%}")
        results_table.add_row("Avg Latency", f"{hybrid_stats['avg_latency']:.2f}s")
        results_table.add_row("Successful Queries", f"{hybrid_stats['n_successful']} / {hybrid_stats['n_queries']}")
        
        console.print(results_table)

    # JSON output for frontend
    if args.json or args.output_file:
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
            "dataset_info": results_data["dataset_info"],
            "dataset_size": results_data["dataset_size"],
            "sample_queries": [
                {
                    "id": s.get("id", ""),
                    "query": s.get("query", ""),
                    "ground_truth": s.get("answer_texts", [""])[0][:100] if s.get("answer_texts") else ""
                } for s in samples[:5]
            ],
            "metrics": {k: serialize_stats(v) for k, v in results_data["metrics"].items()}
        }
        
        if args.json:
            print(json.dumps(serialized_data, indent=2))
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            console.print(f"[bold green]Results saved to {args.output_file}[/bold green]")
    
    # Statistical significance section
    if args.ablation:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Statistical Analysis[/bold cyan]",
            title="📊 Statistics",
            border_style="cyan"
        ))
        
        stats_table = Table(title="Detailed Statistics (Mean ± Std, 95% CI)", box=box.ROUNDED)
        stats_table.add_column("Config", style="magenta")
        stats_table.add_column("Hit Rate", justify="center", style="green")
        stats_table.add_column("Hit Rate 95% CI", justify="center", style="green")
        stats_table.add_column("MRR", justify="center", style="blue")
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


if __name__ == "__main__":
    main()
