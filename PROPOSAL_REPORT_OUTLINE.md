# Untango: Dependency-Aware Agentic Code Assistance via Retrieval-Augmented Generation
**Proposal Report Outline**

---

## 1. Abstract
*   **Context**: Modern AI-powered code assistants (GitHub Copilot, Cursor) are essential but limited by "dependency blindness" (analyzing only local repo code).
*   **Problem**: Lack of knowledge about external dependency implementation details leads to inaccurate advice for library-intensive tasks.
*   **Solution**: **Untango**, a dependency-aware agentic code assistant.
    *   Ingests/indexes both repository and dependency source code.
    *   Uses hybrid retrieval (Vector + BM25).
    *   Implements an agentic reasoning loop (autonomous search/read/reason).
*   **Results**: Significantly outperforms standard RAG in code comprehension; provides richer, dependency-aware responses.

---

## 2. Introduction
> **Rubric Alignment**: "Clear description of the problem domain", "Current state of the art and its limitations", "Significance of the problem", "Target audience".

### 2.1 Motivation
*   LLMs have revolutionized software dev (Copilot, Cursor, Codeium).
*   Standard RAG pipelines index user's local files.
*   **Gap**: Modern software relies heavily on external libraries (dependencies). Existing tools treat these as static code and fail to provide accurate suggestions for newest package versions.
*   **Scenario**: Developer asks about `Session` interaction with `SQLAlchemy`. Current tools guess based on the LLM's training data; Untango looks up the actual SQLAlchemy code.

### 2.2 Problem Statement
*   **Dependency Blindness**: Missing context from imported libraries.
*   **Passive Retrieval**: One-shot retrieval is insufficient for complex exploration.
*   **Limited Reasoning**: Concatenating context without analysis is suboptimal.
*   **Significance**: Accurate code assistance requires full ecosystem visibility, not just local code visibility.

### 2.3 Research Questions
> **Rubric Alignment**: "Clearly articulated research questions", "Hypotheses and assumptions".
*   **RQ1**: Does incorporating dependency source code into the retrieval corpus improve response quality?
    *   *Hypothesis*: Access to dependency internals will reduce hallucinations and improve specificity.
*   **RQ2**: Can an agentic reasoning loop that iteratively retrieves and analyzes code outperform single-pass RAG?
    *   *Hypothesis*: Multi-turn reasoning allows for "fact-checking" and deeper exploration.
*   **RQ3**: What is the optimal combination of vector similarity and keyword-based retrieval for code search?
    *   *Hypothesis*: Hybrid search (Semantic + Exact Match) is superior for code which has specific identifiers.

### 2.4 Contributions
> **Rubric Alignment**: "Description of major contributions", "Technical innovations", "Practical impact".
1.  **Untango System**: Open-source, full-stack code assistant with dependency ingestion.
2.  **Agentic RAG Architecture**: Multi-turn look-think-act loop.
3.  **Hybrid Search Pipeline**: RRF-based combination of Dense + Sparse retrieval.
4.  **Empirical Evaluation**: Quantitative (Hit Rate, MRR) and Qualitative analysis.

---

## 3. Related Work (Literature Review)
> **Rubric Alignment**: "Comprehensive literature review", "Critical analysis", "Integration of multiple perspectives".

### 3.1 Retrieval-Augmented Generation (RAG)
*   Foundational work: **Lewis et al. [4]** (Original RAG).
*   Applications in QA: **Izacard & Grave [5]** (DPR).
*   *Relevance*: Establishes the baseline methodology we improve upon.

### 3.2 Hybrid Search & Information Retrieval
*   Lexical Search: **BM25 [7]** (Exact matching, crucial for code identifiers).
*   Dense Retrieval: **Xu et al. [8]** (DPR, semantic understanding).
*   Fusion: **Reciprocal Rank Fusion (RRF) [9]**.
*   *Critical Analysis*: Pure vector search fails on specific function names; pure BM25 misses "how to..." concepts. Hybrid is required.

### 3.3 LLM-Based Code Assistants
*   Models: **Codex [10]**, **Code Llama [11]**, **StarCoder [12]**.
*   Tools: Copilot, Cursor.
*   *Limitation*: Focus on generation/completion, not deep ecosystem understanding.

### 3.4 Tool-Augmented Language Models
*   Agentic Frameworks: **Toolformer [13]**, **ReAct [14]**.
*   Function Calling: **OpenAI [15]**, **Gemini [16]**.
*   *Application*: We apply ReAct principles to code navigation (Search -> Read -> Think -> Answer).

### 3.5 Dependency Analysis
*   Software Engineering context: **Kikas et al. [17]**, **Decan et al. [18]** (Ecosystem analysis).
*   *Novelty*: Integrating rigorous dependency analysis into the RAG context window.

---

## 4. Methodology
> **Rubric Alignment**: "Detailed experimental setup", "Data collection/preprocessing", "Model architecture", "Training/optimization".

### 4.1 System Architecture
*   **High-Level Design**:
    *   **Frontend**: React-based chat interface.
    *   **Backend**: FastAPI, ChromaDB (Vector Store), Vertex AI (LLM).
    *   **Orchestrator**: Agentic Loop.
*   *(Recommended: Insert System Architecture Diagram here)*

### 4.2 Data Collection & Preprocessing (Ingestion)
*   **Repository Scanning**: `RepoMapper` identifies structure and `requirements.txt`.
*   **Dependency Resolution**: Fetching source code for packages (from `site-packages` or remote).
*   **AST-Based Chunking**:
    *   *Technique*: Using Python `ast` module to split by Function/Class nodes.
    *   *Why*: Avoids breaking code logic in the middle (unlike character-count splitting).
    *   *Metadata*: Enriched with file path, line numbers, scope.

### 4.3 Hybrid Search Implementation
*   **Vector Index**: ChromaDB (Default embedding model: `all-MiniLM-L6-v2` or similar).
*   **Keyword Index**: RAM-based BM25 index on tokenized code.
*   **Fusion Strategy**: Reciprocal Rank Fusion (RRF) with $k=60$.
    *   Formula: $Score(d) = \sum \frac{1}{k + rank(d)}$.

### 4.4 Agentic Reasoning Loop (The "Brain")
*   **Model**: Google Gemini 2.5 Flash (via Vertex AI).
*   **Pattern**: ReAct (Reason + Act).
*   **Tools Provided**:
    *   `rag_search(query)`: Semantic + Keyword search.
    *   `read_file(path)`: Full content inspection.
    *   `get_context_report()`: Env/Dependency info.
*   **Flow**:
    1. User asks question.
    2. Agent analyzes if it needs external info.
    3. Agent calls `rag_search` recursively or reads files.
    4. Agent synthesizes answer.

---

## 5. Experimental Setup
> **Rubric Alignment**: "Quantitative Metrics", "Comparative analysis", "Ablation studies".

### 5.1 Dataset Construction
*   **Curated Queries**: 15 distinct technical queries targeting:
    *   Architectural understanding.
    *   Implementation details.
    *   Dependency usage.
*   **Ground Truth**: Manually identified "Gold Standard" code chunks that contain the answer.

### 5.2 Metrics
1.  **Hit Rate (@ k)**: Percentage of queries where the "Gold" chunk appears in top-k results.
2.  **Mean Reciprocal Rank (MRR)**: Position of the first relevant result (higher is better).
3.  **Latency**: Time to first token / Total completion time (Trade-off metric).

### 5.3 Baselines
*   **Baseline 1**: Vector-only Search.
*   **Baseline 2**: BM25-only Search.
*   **Baseline 3**: Standard RAG (Retrieve-Once-Then-Generate).
*   **Ours**: Untango (Hybrid Search + Agentic Loop).

---

## 6. Results and Analysis
> **Rubric Alignment**: "Clear presentation of findings", "Appropriate visualizations", "Thorough analysis of results".

### 6.1 Quantitative Results (Retrieval Performance)
*   **Table: Ablation Study**
    *   *Show columns*: Method | Hit Rate | MRR | Latency
    *   *Finding*: Hybrid Search outperforms Vector-only (e.g., Hit Rate 93.3% vs 86.7%).
    *   *(Place for Bar Chart: Retrieval Performance Comparison)*

### 6.2 Qualitative Analysis (Agent Performance)
*   **Comparison**: Agentic vs. Standard RAG.
*   **Case Study: `requests` Library**
    *   *Query*: "How does requests handle connection pooling?"
    *   *Standard RAG*: General hallucinations or "I don't know".
    *   *Untango*: Finds usage -> Reads `requests` source -> Finds `urllib3` import -> Explains connection pool logic.
*   **Observation**: Agentic loop allows "Self-Correction" (if search fails, try different query).

### 6.3 Performance & Trade-offs
*   **Latency Cost**: Agentic loop takes 3-4x longer (5-8s vs 2s).
*   **Justification**: For complex engineering queries, accuracy > speed.

---

## 7. Discussion
> **Rubric Alignment**: "Discussion of implications", "Limitations and future work".

### 7.1 Implications for Software Engineering
*   Dependency-aware tools can reduce "context switching" (reading docs vs coding).
*   Potential to automate debugging that spans into library code.

### 7.2 Limitations
*   **Language Support**: Currently Python-centric (AST dependency).
*   **Scalability**: Indexing *all* dependencies is expensive (storage/compute). Need "Selective Ingestion".
*   **Context Window**: Even with large windows, noise from irrelevant dependency code can distract weak models.

### 7.3 Future Work
*   **Multi-language AST**: Support JS/TS, Java.
*   **Selective Ingestion**: Heuristics to index only *used* parts of dependencies.
*   **Graph RAG**: Using call-graphs instead of just text chunks.

---

## 8. Conclusion
*   Summary of Untango: A step towards "holistic" code understanding.
*   Key Achievement: Successfully integrated dependency code into the RAG loop.
*   Final Thought: As LLMs get faster, agentic loops will become the default for code assistance.

---

## 9. References
[1] GitHub Copilot. (2021). https://github.com/features/copilot
[2] Cursor. (2023). https://cursor.sh
[3] Codeium. (2023). https://codeium.com
[4] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
[5] Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models. *EACL*.
[6] Shao, Z., et al. (2023). Enhancing Retrieval-Augmented Large Language Models. *EMNLP*.
[7] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
[8] Xu, L., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP*.
[9] Cormack, G. V., et al. (2009). Reciprocal Rank Fusion. *SIGIR*.
[10] Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.
[11] Rozière, B., et al. (2023). Code Llama: Open Foundation Models for Code.
[12] Li, R., et al. (2023). StarCoder: A State-of-the-Art LLM for Code.
[13] Schick, T., et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. *NeurIPS*.
[14] Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR*.
[15] OpenAI. (2023). Function Calling and Other API Updates.
[16] Google DeepMind. (2023). Gemini: A Family of Highly Capable Multimodal Models.
[17] Kikas, R., et al. (2017). Structure and Evolution of Package Dependency Networks. *MSR*.
[18] Decan, A., et al. (2019). An Empirical Comparison of Dependency Network Evolution. *ESE*.
[19] ChromaDB. (2024). Hybrid Search Documentation.

---

## 10. Appendices
### Appendix A: System Requirements
*   Overview of tech stack (Py3.11, Docker, Chroma, Vertex AI).

### Appendix B: Reproduction Instructions
*   `git clone`
*   `docker-compose up`
*   `python scripts/evaluate.py`

### Appendix C: Evaluation Dataset Details
*   List of queries used.
*   Definition of "Correctness" for metrics.
