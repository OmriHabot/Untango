# Untango: Dependency-Aware Agentic Code Assistance via MCP and Gemini 2.0 Flash

**COMSE6998-015: Introduction to LLM-based Generative AI Systems**
**Fall 2025**
**Project Proposal Report**

---

## 1. Abstract

Modern AI-powered code assistants like GitHub Copilot and Cursor have transformed software development by providing context-aware suggestions and explanations. However, these tools suffer from a critical limitation: "dependency blindness." They analyze only the immediate repository code while ignoring the implementation details of external dependencies, treating them as static black boxes. This leads to inaccurate advice when developers ask questions involving library internals or require knowledge of the newest package versions.

We present **Untango**, a dependency-aware agentic code assistant that addresses this gap. Untango leverages three key innovations:

1. **Model Context Protocol (MCP)**: A standardized protocol for tool discovery and execution, enabling clean separation between the LLM agent and its capabilities
2. **Gemini 2.0 Flash (via Vertex AI)**: Google's state-of-the-art multimodal model with native function calling for reliable agentic tool use
3. **Deep Exploration Workflow**: A mandatory multi-step reasoning process that ensures the agent thoroughly explores codebases before answering

Our qualitative evaluation demonstrates that Untango provides significantly richer, more accurate responses than traditional RAG systems by:
- Reading and synthesizing actual source code rather than relying on embeddings
- Following import chains and call graphs to understand system behavior
- Providing step-by-step instructions verified against actual configuration files

---

## 2. Introduction

### 2.1 Motivation

The emergence of Large Language Models (LLMs) has revolutionized software development workflows. Tools like GitHub Copilot [1], Cursor [2], and Codeium [3] have become indispensable for developers, offering code completion, explanation, and refactoring capabilities. These systems typically employ Retrieval-Augmented Generation (RAG) pipelines that index the user's codebase and retrieve relevant context when answering queries.

However, a fundamental limitation persists: **existing code assistants operate within the confines of the user's repository**, treating external dependencies as black boxes. When a developer asks, "How does the `Session` class in my authentication module interact with `SQLAlchemy`'s connection pooling?", current tools can only examine the developer's code—not SQLAlchemy's implementation. They are forced to "guess" based on the LLM's pre-trained knowledge, which may be outdated or hallucinated.

Furthermore, **passive RAG retrieval fails for complex queries**. A single embedding-based search rarely surfaces all the context needed to understand a multi-component system. Developers exploring new codebases need an assistant that can **actively explore** the code, following imports and reading configuration files, just as they would manually.

### 2.2 Problem Statement

We identify four key limitations in current code assistance tools:

1. **Dependency Blindness**: Tools index only local files, missing critical context from imported libraries. This leads to errors when debugging issues that originate in third-party code.

2. **Passive Retrieval**: Standard RAG pipelines perform one-shot retrieval. If the initial search misses the relevant context, the model fails to answer.

3. **Limited Reasoning**: Retrieved context is directly concatenated to the prompt without strategic analysis. The model cannot "realize" it is missing information and request more.

4. **No Tool Standard**: Each code assistant implements its own proprietary tool interface, making it difficult to extend capabilities or integrate with existing developer infrastructure.

### 2.3 Research Questions

This work addresses the following research questions:

* **RQ1**: Does an agentic architecture that actively explores code outperform passive RAG retrieval?
  * *Hypothesis*: Multi-turn tool-using agents provide more accurate and complete answers by reading actual source files rather than relying on embedding similarity.

* **RQ2**: Can the Model Context Protocol (MCP) provide a clean abstraction for code exploration tools?
  * *Hypothesis*: Standardizing tool interfaces via MCP enables better separation of concerns and easier extensibility.

* **RQ3**: What system prompt design maximizes agent exploration behavior?
  * *Hypothesis*: Mandatory tool-use constraints and deep exploration workflows reduce hallucination and improve answer quality.

### 2.4 Contributions

We make the following contributions:

1. **Untango System**: An open-source, production-ready code assistant featuring an MCP-based tool server, React frontend, and Gemini 2.0 Flash agentic backend.

2. **MCP Tool Server**: A comprehensive set of 11 tools exposed via the Model Context Protocol, including RAG search, file reading, git operations, test discovery, and linting.

3. **Deep Exploration Workflow**: A novel system prompt architecture that mandates thorough codebase exploration before answering, with specific strategies for different repository types.

4. **Qualitative Evaluation Framework**: Comparative case studies demonstrating the superiority of agentic exploration over passive RAG for real-world code understanding tasks.

---

## 3. System Architecture

### 3.1 Overview

Untango consists of four primary components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                           │
│   (Chat UI, Repository Ingestion, Watch Mode, Streaming)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────▼───────────────────────────────────────┐
│                        FastAPI Backend                           │
│     ┌──────────────────────────────────────────────────────┐    │
│     │              MCP Agent (Gemini 2.0 Flash)             │    │
│     │  - Multi-turn reasoning loop (ReAct pattern)          │    │
│     │  - Dynamic tool discovery via MCP                      │    │
│     │  - Streaming response generation                       │    │
│     └────────────────────────┬─────────────────────────────┘    │
│                              │ MCP Protocol                      │
│     ┌────────────────────────▼─────────────────────────────┐    │
│     │                   MCP Tool Server                      │    │
│     │  - rag_search: Hybrid vector + BM25 retrieval         │    │
│     │  - read_file: Source code inspection                   │    │
│     │  - list_files: Directory traversal                     │    │
│     │  - git_status/diff/log: Version control                │    │
│     │  - run_tests/run_linter: Quality checks                │    │
│     │  - find_function_usages: AST-based analysis           │    │
│     └────────────────────────┬─────────────────────────────┘    │
│                              │                                   │
│     ┌────────────────────────▼─────────────────────────────┐    │
│     │                    ChromaDB                            │    │
│     │        (Vector embeddings + BM25 index)               │    │
│     └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Context Protocol (MCP)

MCP is an open protocol that standardizes how AI models interact with external tools and data sources. Untango implements MCP to provide:

**Tool Discovery**: The agent dynamically discovers available tools at runtime by querying the MCP server. This enables adding new capabilities without modifying the agent.

**Type-Safe Execution**: Tools are defined with JSON Schema input specifications, ensuring the LLM provides correctly-typed arguments.

**Clean Separation**: The tool implementations (file I/O, git commands, test runners) are fully decoupled from the LLM agent logic.

Our MCP server exposes 11 tools:

| Tool | Description |
|------|-------------|
| `rag_search(query)` | Hybrid semantic + keyword search across ingested code |
| `read_file(filepath, max_lines)` | Read source file contents |
| `list_files(directory)` | List directory contents with type indicators |
| `find_function_usages(function_name)` | AST-based usage analysis |
| `git_status()` | Current branch and file states |
| `git_diff(filepath)` | Uncommitted changes |
| `git_log(filepath, max_commits)` | Commit history |
| `discover_tests()` | Find pytest test functions |
| `run_tests(test_path, verbose)` | Execute pytest |
| `run_linter(filepath)` | Run ruff/flake8/pylint |
| `get_active_repo_path()` | Get repository filesystem path |

### 3.3 Gemini 2.0 Flash Integration

We use **Gemini 2.0 Flash** via Google Vertex AI as our primary LLM. Key advantages include:

- **Native Function Calling**: Built-in support for structured tool calls with typed arguments
- **Long Context Window**: 1M+ token context for ingesting large codebases
- **Multimodal Capabilities**: Future extensibility for diagram understanding
- **Low Latency**: Optimized for real-time interactive use
- **Cost Efficiency**: $0.30/M input tokens, $2.50/M output tokens

The agent operates in a multi-turn loop with up to 15 tool-use turns per query, allowing deep exploration of complex codebases.

### 3.4 Deep Exploration Workflow

Unlike traditional RAG systems that perform single-shot retrieval, Untango mandates a comprehensive exploration workflow:

**Step 1: Map the Territory**
- Always call `list_files()` at the repository root
- Explore major subdirectories (frontend/, backend/, api/, etc.)
- Check for README files in each component

**Step 2: Identify Key Files**
- Docker configuration (Dockerfile, docker-compose.yml)
- Package managers (package.json scripts, requirements.txt)
- Entry points (main.py, index.js, app.py)
- Environment config (.env.example, settings.py)

**Step 3: Multi-Component Exploration**
- For full-stack projects, explore EACH component separately
- Read actual config files, not just READMEs
- Provide instructions for running each component

**Step 4: Code Following**
- When seeing an import, read the imported module
- Trace function calls to their definitions
- Never guess based on names alone

The system prompt explicitly states: "You are FORBIDDEN from answering questions about the codebase without calling at least one tool."

---

## 4. Implementation Details

### 4.1 Frontend Architecture

The React frontend provides:

- **Repository Ingestion**: Upload local directories via File System Access API or clone from GitHub
- **Watch Mode**: Real-time file change detection with automatic re-sync
- **Watch Status Indicator**: Visual indicators showing which repositories are being actively watched
- **Streaming Chat**: Server-Sent Events (SSE) for real-time token streaming
- **Tool Visualization**: Display of tool calls and results during agent reasoning

### 4.2 Ingestion Pipeline

Our AST-based ingestion pipeline:

1. **Scan**: Traverse directory tree, respecting `.gitignore` and skip patterns
2. **Parse**: Use Python AST to extract functions, classes, and modules
3. **Chunk**: Create semantically meaningful chunks (one per function/class)
4. **Embed**: Generate embeddings using sentence-transformers
5. **Index**: Store in ChromaDB with metadata for hybrid search

### 4.3 Hybrid Search

We combine two retrieval strategies:

- **Vector Search**: Semantic similarity using `all-MiniLM-L6-v2` embeddings
- **BM25 Search**: Keyword matching with code-aware tokenization (handles snake_case, camelCase)
- **Reciprocal Rank Fusion (RRF)**: Merge results with k=60

---

## 5. Qualitative Evaluation

### 5.1 Evaluation Methodology

We evaluate Untango through comparative case studies, demonstrating the qualitative difference between:

1. **Standard RAG**: Single-shot embedding retrieval with direct answer generation
2. **Untango (Agentic)**: Multi-turn exploration with tool-verified answers

For each case study, we document:
- The user query
- The exploration steps taken by the agent
- The final answer quality and accuracy
- Key insights that would be missed by passive retrieval

### 5.2 Case Study 1: "How do I run this project?"

**Query**: "How do I run this project?"

**Standard RAG Response**:
```
Based on the README.md, you can run this project with:

1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Run `python main.py`

This will start the application.
```

**Untango Response** (after 8 tool calls):
```
This is a **full-stack application** with separate frontend and backend components.

## Backend (Python + FastAPI)

The backend runs via Docker Compose:

```bash
cd /path/to/project
docker-compose up --build -d
```

This starts:
- `rag-backend` on port 8001 (FastAPI with Gemini integration)
- `chromadb-server` on port 8000 (Vector database)

## Frontend (React + Vite)

```bash
cd frontend
pnpm install
pnpm run dev
```

The frontend runs on http://localhost:5173

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GOOGLE_CLOUD_LOCATION`: Region (default: "global")

## Verification

1. Backend health: `curl http://localhost:8001/health`
2. Open frontend at http://localhost:5173
3. Ingest a repository using the "+" button
```

**Analysis**: The standard RAG response relied on outdated README content. Untango:
- Explored both root and subdirectories
- Read `docker-compose.yml` to understand the container architecture
- Read `frontend/package.json` to find the correct scripts
- Read `.env.example` to identify required configuration
- Provided step-by-step verified instructions

### 5.3 Case Study 2: "How does authentication work?"

**Query**: "How does authentication work in this codebase?"

**Standard RAG Response**:
```
Based on the search results, there appears to be an authentication module in 
`api/auth.py`. The code uses JWT tokens for authentication. Users authenticate 
by providing credentials which are validated against the database.
```

**Untango Response** (after 12 tool calls):
```
## Authentication Architecture

This application uses **Vertex AI's built-in authentication** rather than 
implementing custom JWT/session authentication.

### How It Works

1. **GCP Service Account**: The backend authenticates to Google Cloud using 
   application default credentials or a service account key.

2. **No User Authentication**: The current implementation does not have 
   user-level authentication. All requests are processed under the service 
   account's permissions.

3. **Active Repository State**: The only "session" concept is `active_repo_state` 
   in `app/active_repo_state.py`:
   
   ```python
   class ActiveRepoState:
       _active_repo_id: str = "default"
       _ingestion_statuses: Dict[str, str] = {}
   ```
   
   This tracks which repository is currently selected, not user identity.

### Security Considerations

- The API is exposed on port 8001 without API key validation
- All endpoints are publicly accessible once the container is running
- Chat history is stored per-repository, not per-user

### If You Need User Auth

To add user authentication, you would need to:
1. Add an auth middleware to FastAPI
2. Implement token validation (e.g., Firebase Auth, Auth0)
3. Associate repositories and chat history with user IDs
```

**Analysis**: The standard RAG response hallucinated a JWT system that doesn't exist. Untango:
- Searched for "authentication" and found no user auth module
- Read the actual `active_repo_state.py` to understand the session model
- Examined FastAPI endpoints for auth middleware (found none)
- Provided accurate assessment with actionable recommendations

### 5.4 Case Study 3: "What happens when I send a chat message?"

**Query**: "Trace what happens when a user sends a chat message"

**Standard RAG Response**:
```
When a user sends a chat message:
1. The frontend sends a POST request to the chat endpoint
2. The backend processes the message using RAG to find relevant context
3. The response is generated and sent back to the user
```

**Untango Response** (after 15 tool calls):
```
## Complete Chat Message Flow

### 1. Frontend (React)

The chat input is handled in `frontend/src/components/ChatWindow.tsx`:

```typescript
const handleSubmit = async (e: FormEvent) => {
  const response = await api.chat(messages);
  // Streaming handled via useChatStore
};
```

### 2. API Client

`frontend/src/api/client.ts` sends to `/chat-stream`:

```typescript
chatStream: (messages: Message[]) => {
  return fetch(`${API_URL}/chat-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  });
}
```

### 3. FastAPI Backend

`app/main.py` routes to the streaming endpoint:

```python
@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    # Trigger smart ingestion sync
    await current_ingest_manager.sync_repo()
    
    # Save user message to history
    chat_history_manager.add_message(active_repo_id, request.messages[-1])
    
    # Stream response
    return StreamingResponse(
        chat_with_mcp_agent_stream(request),
        media_type="application/x-ndjson"
    )
```

### 4. MCP Agent

`app/agents/mcp_agent.py` orchestrates the response:

1. **Context Preparation**: Loads repository context, README, file tree
2. **Tool Discovery**: Fetches available tools from MCP server
3. **Multi-Turn Loop** (up to 15 turns):
   - Model generates thought + tool call
   - MCP server executes tool
   - Result fed back to model
4. **Streaming**: Tokens yielded as NDJSON events

### 5. Tool Execution

`app/mcp_server.py` handles tool calls:

```python
@mcp.tool()
def read_file(filepath: str, max_lines: int = 500) -> str:
    repo_path = _get_active_repo_path()
    full_path = os.path.join(repo_path, filepath)
    # Read and return file content
```

### 6. Response Assembly

The streaming response format:
- `{"type": "token", "content": "..."}`
- `{"type": "tool_start", "tool": "read_file", "args": {...}}`
- `{"type": "tool_end", "tool": "read_file", "result": "..."}`
- `{"type": "usage", "usage": {...}}`
```

**Analysis**: The standard RAG response was generic and unhelpful. Untango:
- Traced the complete flow from frontend to backend
- Read actual source code at each layer
- Provided code snippets from the real implementation
- Documented the streaming protocol and data format

### 5.5 Evaluation Summary

| Aspect | Standard RAG | Untango (Agentic) |
|--------|-------------|-------------------|
| **Answer Accuracy** | Often outdated or hallucinated | Verified against actual code |
| **Completeness** | Shallow, single-topic | Deep, follows connections |
| **Actionability** | Generic instructions | Copy-paste commands |
| **Code References** | Vague file mentions | Specific line excerpts |
| **Multi-Component** | Often misses components | Explores all subdirectories |
| **Tool Calls** | 0 (static context) | 8-15 per query |

---

## 6. Discussion

### 6.1 Why Agentic > RAG for Code

Traditional RAG assumes that relevant context can be identified via embedding similarity. This assumption fails for code because:

1. **Code requires tracing**: Understanding behavior requires following imports and call chains
2. **Names are misleading**: A file named `auth.py` might not exist, but auth logic could be in `api/middleware.py`
3. **Config is scattered**: Running instructions exist in docker-compose.yml, package.json, and .env.example
4. **Code evolves**: READMEs become outdated faster than actual implementations

The agentic approach addresses these by **reading actual files** rather than relying on embeddings to find "similar" content.

### 6.2 MCP as an Abstraction Layer

The Model Context Protocol provides several benefits:

1. **Extensibility**: Adding a new tool requires only implementing the MCP interface
2. **Testing**: Tools can be tested independently of the LLM
3. **Portability**: The same tools can be used with different LLM providers
4. **Caching**: Tool schema can be cached, reducing latency

### 6.3 Gemini 2.0 Flash Performance

We chose Gemini 2.0 Flash for its:

- **Reliability**: Native function calling reduces parsing errors compared to text-based tool use
- **Context Length**: 1M+ tokens enables full codebase context when needed
- **Speed**: Low latency enables interactive exploration
- **Cost**: ~$0.001-0.01 per complex query

### 6.4 Limitations

1. **Latency**: Multi-turn exploration adds ~5-15 seconds per query
2. **Token Costs**: Deep exploration uses more tokens than single-shot RAG
3. **Python Focus**: AST chunking currently supports Python only
4. **No Execution Sandbox**: Cannot safely execute arbitrary code

### 6.5 Future Work

1. **Selective Dependency Ingestion**: Index only public APIs of dependencies
2. **Multi-Language Support**: TypeScript, Java, Go AST parsers
3. **Code Execution**: Sandboxed execution for validation
4. **Graph RAG**: Represent codebase as call graph for multi-hop retrieval

---

## 7. Conclusion

We presented Untango, a system that demonstrates the superiority of **agentic code exploration** over passive RAG for code assistance. By combining:

- **Model Context Protocol** for standardized tool interfaces
- **Gemini 2.0 Flash** for reliable function calling
- **Deep Exploration Workflow** for thorough codebase understanding

Untango achieves significantly higher answer quality and accuracy than traditional retrieval-based systems.

Our qualitative evaluation shows that:
- **Standard RAG often hallucinates** or provides outdated information
- **Agentic exploration verifies claims** against actual source code
- **Multi-component projects require systematic exploration** of all subdirectories
- **Reading actual config files** is essential for accurate setup instructions

As software systems grow in complexity, tools like Untango that can actively navigate codebases will become essential for developer productivity.

---

## 8. References

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
[15] Model Context Protocol. (2024). https://modelcontextprotocol.io
[16] Google DeepMind. (2024). Gemini 2.0: A New Era of Multimodal AI.
[17] Kikas, R., et al. (2017). Structure and Evolution of Package Dependency Networks. *MSR*.
[18] Decan, A., et al. (2019). An Empirical Comparison of Dependency Network Evolution. *ESE*.

---

## Appendix A: System Requirements

- **Language**: Python 3.11+, TypeScript/React
- **Containerization**: Docker & Docker Compose
- **Database**: ChromaDB (Vector Store)
- **LLM Provider**: Google Vertex AI (Gemini 2.0 Flash)
- **Frontend**: React, Vite, Zustand

## Appendix B: Reproduction Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/omrihabot/untango.git
   cd untango
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your GCP project ID
   ```

3. Start backend services:
   ```bash
   docker-compose up --build -d
   ```

4. Start frontend:
   ```bash
   cd frontend
   pnpm install
   pnpm run dev
   ```

5. Open http://localhost:5173

## Appendix C: MCP Tool Schemas

```json
{
  "tools": [
    {
      "name": "rag_search",
      "description": "Search the codebase using hybrid vector + keyword search",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
      }
    },
    {
      "name": "read_file",
      "description": "Read file content from the repository",
      "inputSchema": {
        "type": "object",
        "properties": {
          "filepath": {"type": "string"},
          "max_lines": {"type": "integer", "default": 500}
        },
        "required": ["filepath"]
      }
    }
  ]
}
```

## Appendix D: Sample Agent Trace

```json
{
  "query": "How do I run the tests?",
  "turns": [
    {"tool": "list_files", "args": {"directory": "."}, "files_found": 15},
    {"tool": "list_files", "args": {"directory": "tests"}, "files_found": 8},
    {"tool": "read_file", "args": {"filepath": "pyproject.toml"}, "lines": 45},
    {"tool": "discover_tests", "args": {}, "tests_found": 12},
    {"tool": "read_file", "args": {"filepath": "tests/test_search.py"}, "lines": 89}
  ],
  "final_answer": "Run tests with: `pytest tests/ -v`"
}
```
