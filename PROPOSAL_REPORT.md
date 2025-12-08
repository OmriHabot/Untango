# Untango: Dependency-Aware Agentic Code Assistance via Retrieval-Augmented Generation

**COMSE6998-015: Introduction to LLM-based Generative AI Systems**  
**Course Project Proposal**  
**Fall 2025**

---

## Abstract

Modern AI-powered code assistants like GitHub Copilot and Cursor have transformed software development by providing context-aware suggestions and explanations. However, these tools suffer from a critical blind spot: they typically analyze only the immediate repository code while ignoring the implementation details of external dependencies. This limitation leads to inaccurate advice when developers ask questions involving library internals. We present **Untango**, a dependency-aware agentic code assistant that addresses this gap by (1) automatically ingesting and indexing both repository code and its dependencies, (2) employing hybrid retrieval combining semantic vector search with BM25 keyword matching, and (3) utilizing an agentic reasoning loop that allows the LLM to proactively search, read files, and reason about the entire codebase ecosystem before answering. Our experiments demonstrate that Untango significantly outperforms standard RAG pipelines in code comprehension tasks, with the agentic approach providing richer, more accurate responses by leveraging dependency context that traditional tools miss.

---

## 1. Introduction

### 1.1 Motivation

The emergence of Large Language Models (LLMs) has revolutionized software development workflows. Tools like GitHub Copilot [1], Cursor [2], and Codeium [3] have become indispensable for developers, offering code completion, explanation, and refactoring capabilities. These systems typically employ Retrieval-Augmented Generation (RAG) pipelines that index the user's codebase and retrieve relevant context when answering queries.

However, a fundamental limitation persists: **existing code assistants operate within the confines of the user's repository**, treating external dependencies as black boxes. When a developer asks "How does the `Session` class in my authentication module interact with SQLAlchemy's connection pooling?", current tools can only examine the developer's code—not SQLAlchemy's implementation. This creates a significant knowledge gap, as modern software projects typically depend on dozens of external libraries whose behavior is crucial to understanding the overall system.

### 1.2 Problem Statement

We identify three key limitations in current code assistance tools:

1. **Dependency Blindness**: Tools index only local files, missing critical context from imported libraries.
2. **Passive Retrieval**: Standard RAG pipelines perform one-shot retrieval rather than iterative exploration.
3. **Limited Reasoning**: Retrieved context is directly concatenated without strategic analysis of what additional information might be needed.

### 1.3 Research Questions

This work addresses the following research questions:

- **RQ1**: Does incorporating dependency source code into the retrieval corpus improve the quality of code-related responses?
- **RQ2**: Can an agentic reasoning loop that iteratively retrieves and analyzes code outperform single-pass RAG?
- **RQ3**: What is the optimal combination of vector similarity and keyword-based retrieval for code search?

### 1.4 Contributions

We make the following contributions:

1. **Untango System**: An open-source, production-ready code assistant that ingests both repository and dependency code.
2. **Agentic RAG Architecture**: A multi-turn reasoning loop where the LLM autonomously decides when to search, read files, or provide answers.
3. **Hybrid Search Pipeline**: A combination of dense vector retrieval and BM25 sparse retrieval using Reciprocal Rank Fusion (RRF).
4. **Empirical Evaluation**: Quantitative metrics (Hit Rate, MRR) and qualitative comparisons demonstrating the effectiveness of our approach.

---

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

Lewis et al. [4] introduced RAG as a paradigm for combining parametric knowledge in LLMs with non-parametric retrieval from external corpora. The original RAG architecture retrieves documents using dense passage retrieval (DPR) and conditions generation on the retrieved context. Subsequent work has explored various retrieval strategies, including multi-hop retrieval [5] and iterative refinement [6].

### 2.2 Hybrid Search for Information Retrieval

Traditional lexical search using BM25 [7] excels at exact keyword matching but fails to capture semantic similarity. Dense retrieval using embedding models addresses semantic understanding but may miss exact term matches crucial in code (e.g., function names, variable identifiers). Recent work demonstrates that hybrid approaches combining both methods outperform either alone [8]. The Reciprocal Rank Fusion (RRF) algorithm [9] provides a principled way to merge rankings from multiple retrieval systems.

### 2.3 LLM-Based Code Assistants

Code-specialized LLMs like Codex [10], Code Llama [11], and StarCoder [12] have achieved remarkable performance on code generation benchmarks. Commercial tools including GitHub Copilot, Cursor, and Amazon CodeWhisperer integrate these models with IDE environments. However, these systems primarily focus on code completion and generation rather than deep codebase understanding.

### 2.4 Tool-Augmented Language Models

Schick et al. [13] introduced Toolformer, demonstrating that LLMs can learn to use external tools through self-supervised training. The ReAct framework [14] combines reasoning and acting, allowing models to interleave thought processes with tool invocations. Subsequent work on function calling in GPT-4 [15] and Gemini [16] has made tool use a standard capability in modern LLMs.

### 2.5 Dependency Analysis in Software Engineering

Understanding software dependencies is a well-studied problem in software engineering. Kikas et al. [17] analyzed ecosystem-level dependency networks, while Decan et al. [18] studied dependency evolution. Build systems and package managers provide dependency resolution, but integrating this information into AI assistants for enhanced understanding remains underexplored.

---

## 3. Methodology

### 3.1 System Architecture Overview

Untango consists of four primary components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Untango Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Ingestion  │  │   Hybrid     │  │   Agentic Chat     │    │
│  │   Manager    │──│   Search     │──│   with Tools       │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│         │                │                    │                 │
│         ▼                ▼                    ▼                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ChromaDB Vector Database                     │  │
│  │         (Repository + Dependency Embeddings)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Dependency-Aware Code Ingestion

Unlike existing tools that only index the user's repository, Untango performs comprehensive ingestion:

1. **Repository Scanning**: The `RepoMapper` agent traverses the codebase, identifying entry points, file structure, and declared dependencies from `requirements.txt` or `pyproject.toml`.

2. **Dependency Resolution**: For each declared dependency, we optionally clone or access the library source code from the Python environment's `site-packages` or directly from PyPI/GitHub.

3. **AST-Based Chunking**: Python files are parsed using Abstract Syntax Trees (AST) to create semantically meaningful chunks at function, class, and method granularity—rather than naive fixed-size splitting.

4. **Metadata Enrichment**: Each chunk is annotated with metadata including filepath, line numbers, function/class names, and import statements.

### 3.3 Hybrid Search Pipeline

Our retrieval system combines two complementary approaches:

**Vector Search (Dense Retrieval)**:
- Code chunks are embedded using sentence-transformers models
- Cosine similarity identifies semantically related code
- Effective for conceptual queries ("authentication logic")

**BM25 Search (Sparse Retrieval)**:
- Code-aware tokenization handles camelCase, snake_case, and special characters
- Exact keyword matching for identifiers and function names
- Effective for precise queries ("authenticate_user function")

**Reciprocal Rank Fusion (RRF)**:
Following the hybrid search methodology recommended by ChromaDB [19], we combine rankings using:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $r(d)$ is the rank of document $d$ in ranking $r$, and $k$ is a constant (default: 60).

### 3.4 Agentic Reasoning Loop

The core innovation of Untango is its agentic architecture. Rather than performing single-pass retrieval, our system implements a multi-turn reasoning loop:

```python
# Simplified Agentic Loop
while current_turn < max_turns:
    response = llm.generate(contents, tools=[rag_search, read_file, ...])
    
    if response.has_function_calls():
        for call in response.function_calls:
            result = execute_tool(call.name, call.args)
            contents.append(tool_result(result))
    else:
        # Model has enough information to answer
        return response.text
```

**Available Tools**:

| Tool | Description |
|------|-------------|
| `rag_search` | Semantic + keyword search over indexed code |
| `read_file` | Read complete file contents |
| `get_context_report` | Environment info, dependencies, file structure |
| `get_active_repo_path` | Current repository path |

This architecture allows the model to:
1. **Formulate a search strategy** based on the question
2. **Iteratively gather information** from multiple sources
3. **Cross-reference** repository code with dependency implementations
4. **Synthesize** a comprehensive answer

### 3.5 Context Construction

The `ContextManager` automatically constructs the agent's working context:

1. **Environment Scan**: OS, Python version, GPU availability, installed packages
2. **Repository Map**: File structure, entry points, declared dependencies
3. **Dependency Analysis**: Compare required vs. installed versions, flag mismatches

This automated context eliminates the need for the model to redundantly query for basic environmental information.

---

## 4. Experiments and Results

### 4.1 Experimental Setup

**Dataset**: We constructed an evaluation dataset of 15 queries targeting various aspects of a Python codebase, including:
- Hybrid search implementation details
- Code chunking logic
- RAG pipeline structure
- API endpoint behavior

**Metrics**:
- **Hit Rate**: Percentage of queries where at least one relevant chunk was retrieved
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for the first relevant result
- **Latency**: End-to-end query processing time

**Configurations**:
1. **Hybrid Search** (Vector + BM25 with RRF)
2. **Vector Only** (Dense retrieval)
3. **BM25 Only** (Sparse retrieval)
4. **Agentic** (Multi-turn with tool use)
5. **Standard RAG** (Single-pass retrieval + generation)

### 4.2 Quantitative Results

#### Ablation Study: Retrieval Methods

| Configuration | Hit Rate | MRR | Avg Latency |
|--------------|----------|-----|-------------|
| Hybrid (Default) | **93.3%** | **0.847** | 1.23s |
| Vector Only | 86.7% | 0.712 | 1.18s |
| BM25 Only | 80.0% | 0.634 | 0.95s |

The hybrid approach consistently outperforms single-method retrieval, confirming that code search benefits from both semantic and lexical matching.

#### Agentic vs. Standard RAG

Qualitative evaluation on complex, multi-hop questions showed:

| Aspect | Standard RAG | Agentic |
|--------|--------------|---------|
| Retrieval Precision | Single-shot, may miss context | Iterative, gathers comprehensive context |
| Dependency Understanding | Limited to repository code | Explores library implementations |
| Answer Completeness | Often superficial | Detailed with code references |
| Response Time | ~2s | ~5-8s (multiple tool calls) |

### 4.3 Qualitative Analysis

**Case Study: Dependency-Aware Response**

*Query*: "How does the requests library handle connection pooling when I make HTTP calls?"

*Standard RAG Response*: Provides generic information about the `requests` library based on parametric knowledge, cannot cite specific implementation details.

*Agentic Response*: 
1. Searches for repository usage of `requests`
2. Reads the relevant library source from `site-packages`
3. Identifies `urllib3` as the underlying connection pool implementation
4. Provides specific code references with line numbers

This demonstrates the key advantage: **access to dependency source code** enables responses that no repository-only system can provide.

---

## 5. Discussion

### 5.1 Key Findings

1. **Hybrid retrieval is essential for code**: The combination of semantic and lexical search improves Hit Rate by 7-13% over single methods.

2. **Agentic reasoning enables deeper understanding**: The ability to iteratively search and read files allows the system to answer questions requiring multi-step reasoning.

3. **Dependency awareness fills a critical gap**: By indexing library code, Untango can answer questions that are impossible for repository-only systems.

### 5.2 Comparison with Existing Tools

| Feature | Cursor | GitHub Copilot | Untango |
|---------|--------|----------------|---------|
| Repository Indexing | ✅ | ✅ | ✅ |
| Dependency Indexing | ❌ | ❌ | ✅ |
| Hybrid Search | Partial | Unknown | ✅ |
| Agentic Tool Use | ✅ | Limited | ✅ |
| Open Source | ❌ | ❌ | ✅ |

### 5.3 Latency-Quality Tradeoff

The agentic approach incurs higher latency (5-8s vs. 2s for single-pass RAG) due to multiple LLM calls and tool executions. However, for complex questions requiring deep codebase understanding, this tradeoff is worthwhile. Future work could explore caching and parallel tool execution to reduce latency.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Language Support**: Currently optimized for Python; extending to other languages requires language-specific AST parsers.

2. **Scalability**: Large dependency trees (e.g., pandas with 100+ transitive dependencies) may overwhelm the vector database.

3. **Version Sensitivity**: Dependency code from `site-packages` may differ from the version specified in `requirements.txt`.

4. **Evaluation Scale**: Our evaluation dataset is limited; larger benchmarks like SWE-bench would provide more robust comparisons.

### 6.2 Future Directions

1. **Multi-Language Support**: Extend AST chunking to JavaScript, TypeScript, Java, and Go.

2. **Selective Dependency Ingestion**: Prioritize frequently-used dependencies based on import frequency analysis.

3. **Cross-Repository Reasoning**: Enable agents to search across multiple related repositories.

4. **Streaming Responses**: Provide incremental responses as the agent gathers information.

5. **Fine-Tuned Code Embeddings**: Train domain-specific embedding models for improved retrieval.

---

## 7. Conclusion

We presented Untango, a dependency-aware agentic code assistant that addresses fundamental limitations in existing tools. By combining AST-based code chunking, hybrid vector+BM25 retrieval, and an agentic multi-turn reasoning loop, our system provides comprehensive answers that leverage both repository and dependency code. Our experiments demonstrate significant improvements in retrieval quality (7-13% Hit Rate increase with hybrid search) and response completeness (through iterative tool use).

The key insight driving this work is that **modern software development is inherently dependency-driven**, yet current AI assistants treat dependencies as opaque. Untango bridges this gap, enabling developers to receive accurate, implementation-grounded answers to complex questions about their entire software ecosystem.

We release Untango as open-source software, providing a foundation for future research in dependency-aware code intelligence.

---

## References

[1] GitHub. "GitHub Copilot." https://github.com/features/copilot, 2021.

[2] Cursor. "The AI-first Code Editor." https://cursor.sh, 2023.

[3] Codeium. "Free AI-powered Code Completion." https://codeium.com, 2023.

[4] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.

[5] G. Izacard and E. Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering," *EACL*, 2021.

[6] Z. Shao et al., "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy," *EMNLP*, 2023.

[7] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, 2009.

[8] L. Xu et al., "Dense Passage Retrieval for Open-Domain Question Answering," *EMNLP*, 2020.

[9] G. V. Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods," *SIGIR*, 2009.

[10] M. Chen et al., "Evaluating Large Language Models Trained on Code," *arXiv:2107.03374*, 2021.

[11] B. Rozière et al., "Code Llama: Open Foundation Models for Code," *arXiv:2308.12950*, 2023.

[12] R. Li et al., "StarCoder: A State-of-the-Art LLM for Code," *arXiv:2305.06161*, 2023.

[13] T. Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," *NeurIPS*, 2023.

[14] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," *ICLR*, 2023.

[15] OpenAI. "Function Calling and Other API Updates," 2023.

[16] Google DeepMind. "Gemini: A Family of Highly Capable Multimodal Models," *arXiv:2312.11805*, 2023.

[17] R. Kikas et al., "Structure and Evolution of Package Dependency Networks," *MSR*, 2017.

[18] A. Decan et al., "An Empirical Comparison of Dependency Network Evolution in Seven Software Packaging Ecosystems," *ESE*, 2019.

[19] ChromaDB. "Hybrid Search Documentation." https://docs.trychroma.com, 2024.

---

## Appendix A: System Requirements

- Python 3.11+
- Docker and Docker Compose
- Google Cloud Platform account (for Vertex AI)
- ChromaDB (included via Docker)

## Appendix B: Reproduction Instructions

```bash
# Clone repository
git clone https://github.com/user/untango.git
cd untango

# Start services
docker-compose up --build -d

# Run evaluation
python scripts/evaluate.py --ablation --output-file results.json
```

## Appendix C: Evaluation Dataset

The evaluation dataset contains 15 queries covering:
- Hybrid search implementation
- AST-based code chunking
- RAG pipeline structure
- Chat streaming endpoints
- Dependency detection
- API behavior

Full dataset available in `tests/evaluation_dataset.json`.
