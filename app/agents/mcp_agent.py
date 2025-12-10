"""
MCP-Native Chat Agent
Handles user queries using Vertex AI with function calling via MCP server.

This module provides the chat agent functionality using:
- MCP (Model Context Protocol) for tool schemas and execution
- Vertex AI for LLM inference
- Full streaming support for real-time responses
"""
import logging
import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException
from google import genai
from google.genai import types

from ..models import ChatRequest, ChatResponse, TokenUsage, ToolCall
from ..active_repo_state import active_repo_state
from ..repo_manager import repo_manager
from ..context_manager import context_manager

# MCP Client imports
try:
    from mcp import ClientSession, types as mcp_types
    from mcp.client.streamable_http import streamablehttp_client
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============= Configuration =============

# MCP server URL
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")

# Pricing for gemini-3.0-flash (per million tokens)
INPUT_PRICE_PER_MILLION = 0.30  # $0.30 per million tokens
OUTPUT_PRICE_PER_MILLION = 2.50  # $2.50 per million tokens

# MCP tools cache
_cached_mcp_tools: Optional[types.Tool] = None
_mcp_tools_cache_time: float = 0
MCP_TOOLS_CACHE_TTL = 60  # Cache tools for 60 seconds


# ============= Helper Functions =============

def calculate_cost(input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """Calculate cost based on token usage.
    Returns: (input_cost, output_cost, total_cost) in USD"""
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost


def get_active_repo_path() -> str:
    """Helper to get the path of the active repository."""
    repo_id = active_repo_state.get_active_repo_id()
    if repo_id and repo_id != "default":
        return os.path.join(repo_manager.repos_base_path, repo_id)
    return "."


def get_readme_content() -> str:
    """Read README.md from the active repository root if it exists."""
    try:
        repo_path = get_active_repo_path()
        readme_path = os.path.join(repo_path, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Limit to first 1500 chars to encourage tool exploration
                # if len(content) > 1500:
                #     content = content[:1500] + "\n\n... (README truncated - use read_file('README.md') for full content)"
                return content
    except Exception as e:
        logger.warning(f"Failed to read README.md: {e}")
    return ""


def get_repo_file_tree(max_entries: int = 100) -> str:
    """
    Get a breadth-first listing of files and directories in the repository.
    Returns a formatted string with level, type (file/dir), and name.
    Files are prioritized over directories at each level.
    """
    try:
        repo_path = get_active_repo_path()
        if not os.path.isdir(repo_path):
            return ""
        
        # BFS with (path, level) tuples
        from collections import deque
        
        entries = []  # List of (level, is_file, name, relative_path)
        queue = deque([(repo_path, 0)])  # (path, level)
        
        # Directories to skip
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                     '.env', 'dist', 'build', '.next', '.cache', '.pytest_cache',
                     'egg-info', '.eggs', '.tox', '.mypy_cache', '.ruff_cache'}
        
        while queue and len(entries) < max_entries * 2:  # Collect more, then sort/filter
            current_path, level = queue.popleft()
            
            try:
                items = os.listdir(current_path)
            except PermissionError:
                continue
            
            # Separate files and directories
            files = []
            dirs = []
            
            for item in items:
                if item.startswith('.') and item not in {'.env.example', '.gitignore'}:
                    continue  # Skip hidden files except useful ones
                    
                full_path = os.path.join(current_path, item)
                rel_path = os.path.relpath(full_path, repo_path)
                
                if os.path.isfile(full_path):
                    files.append((level, True, item, rel_path))
                elif os.path.isdir(full_path):
                    if item not in skip_dirs and not item.endswith('.egg-info'):
                        dirs.append((level, False, item, rel_path))
                        queue.append((full_path, level + 1))
            
            # Add files first (priority), then dirs
            entries.extend(sorted(files, key=lambda x: x[2].lower()))
            entries.extend(sorted(dirs, key=lambda x: x[2].lower()))
        
        # Sort by level, then files before dirs, then alphabetically
        entries.sort(key=lambda x: (x[0], not x[1], x[2].lower()))
        
        # Truncate to max_entries
        truncated = len(entries) > max_entries
        entries = entries[:max_entries]
        
        # Format output - simple format to avoid confusing function calling
        lines = ["FILE TREE:"]
        for level, is_file, name, rel_path in entries:
            indent = "  " * level
            marker = "f" if is_file else "d"
            lines.append(f"{indent}{marker}: {name}")
        
        if truncated:
            lines.append(f"(truncated at {max_entries} entries)")
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.warning(f"Failed to generate file tree: {e}")
        return ""


# ============= MCP Client Functions =============

async def get_mcp_tools_as_vertex() -> types.Tool:
    """
    Fetch available tools from the MCP server and convert them to Vertex AI format.
    Returns a Tool object with FunctionDeclarations for each MCP tool.
    """
    if not MCP_CLIENT_AVAILABLE:
        logger.warning("MCP client not available, returning empty tool list")
        return types.Tool(function_declarations=[])
    
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # List available tools from MCP server
                tools_result = await session.list_tools()
                
                function_declarations = []
                for tool in tools_result.tools:
                    # Convert MCP tool schema to Vertex AI FunctionDeclaration
                    properties = {}
                    required = []
                    
                    if tool.inputSchema and 'properties' in tool.inputSchema:
                        for prop_name, prop_schema in tool.inputSchema['properties'].items():
                            prop_type = prop_schema.get('type', 'string').upper()
                            # Map JSON schema types to Vertex AI types
                            type_mapping = {
                                'STRING': types.Type.STRING,
                                'INTEGER': types.Type.INTEGER,
                                'NUMBER': types.Type.NUMBER,
                                'BOOLEAN': types.Type.BOOLEAN,
                                'ARRAY': types.Type.ARRAY,
                                'OBJECT': types.Type.OBJECT,
                            }
                            vertex_type = type_mapping.get(prop_type, types.Type.STRING)
                            
                            properties[prop_name] = types.Schema(
                                type=vertex_type,
                                description=prop_schema.get('description', '')
                            )
                        
                        required = tool.inputSchema.get('required', [])
                    
                    func_decl = types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or f"Tool: {tool.name}",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties=properties,
                            required=required
                        )
                    )
                    function_declarations.append(func_decl)
                    logger.debug(f"Loaded MCP tool: {tool.name}")
                
                tool_names = [fd.name for fd in function_declarations]
                logger.info(f"Loaded {len(function_declarations)} tools from MCP server: {tool_names}")
                
                if len(function_declarations) == 0:
                    logger.warning("No tools loaded from MCP server - model will not be able to use tools!")
                
                return types.Tool(function_declarations=function_declarations)
                
    except Exception as e:
        logger.error(f"Failed to fetch MCP tools: {e}", exc_info=True)
        return types.Tool(function_declarations=[])


async def get_cached_mcp_tools() -> types.Tool:
    """Get MCP tools with caching."""
    global _cached_mcp_tools, _mcp_tools_cache_time
    
    now = time.time()
    if _cached_mcp_tools is None or (now - _mcp_tools_cache_time) > MCP_TOOLS_CACHE_TTL:
        _cached_mcp_tools = await get_mcp_tools_as_vertex()
        _mcp_tools_cache_time = now
    
    return _cached_mcp_tools


async def execute_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool via MCP client.
    Connects to the MCP server and calls the specified tool.
    """
    if not MCP_CLIENT_AVAILABLE:
        raise RuntimeError("MCP client not available")
    
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # Call the tool
                result = await session.call_tool(tool_name, arguments=arguments)
                
                # Extract text content from result
                if result.content:
                    for content_block in result.content:
                        if hasattr(content_block, 'text'):
                            return content_block.text
                        elif hasattr(content_block, 'data'):
                            return str(content_block.data)
                
                # Check structured content
                if result.structuredContent:
                    return json.dumps(result.structuredContent)
                
                return "Tool executed successfully (no output)"
                
    except Exception as e:
        # Check for ExceptionGroup (Python 3.11+) or similar wrappers
        if hasattr(e, 'exceptions'):
            for sub_exc in e.exceptions:
                logger.error(f"MCP sub-exception: {sub_exc}")
        logger.error(f"MCP tool execution failed: {e}")
        raise


# ============= System Prompt =============

def get_system_instruction(context_str: str, repo_name: str) -> str:
    """Generate the system instruction with context and CoT protocol."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    readme_content = get_readme_content()
    readme_section = ""
    if readme_content:
        readme_section = f"""
=== PROJECT README.md (ROOT) ===
{readme_content}
=== END README.md ===

> **WARNING**: README files are often outdated. Treat the README as a hint or starting point, NOT as the source of truth.
> The actual source of truth is ALWAYS the code itself. Verify any claims from the README by reading the actual implementation files.
> If the README conflicts with what you observe in the code, trust the code.
"""

    # Get file tree for the repository
    file_tree = get_repo_file_tree(max_entries=50)
    file_tree_section = ""
    if file_tree:
        file_tree_section = f"""
{file_tree}

Use this file tree to understand the repository structure at a glance.
For deeper exploration, use `list_files()` on specific directories.
"""

    return f"""You are an expert software developer and coding assistant for the proprietary **{repo_name}** repository (you have no prior knowledge about this repository).
You have full access to the source code and tools, and you answer questions strictly based on this codebase.

Current date: {current_date}

{file_tree_section}

## Context

The following context was precomputed for you. Use it, but verify details in the actual code when needed.

{context_str}

---

## Step 0: Identify repository type

Your first action must be: call `list_files()` at the repository root.

Use the directory listing to classify the repo into one of:

- Full-stack app: frontend + backend + possible database integration.
- Backend service: APIs, business logic, database schemas, no frontend.
- Frontend app: UI components and client logic, no backend code.
- Python package: library-style structure, exports, `setup.py` / `pyproject.toml`.
- Scripts collection: standalone scripts or CLIs.

After `list_files()`:
- Decide which category best fits.
- Adapt your exploration strategy to that category.
- Avoid guessing architecture based only on names; always confirm by reading files.

---

## Tools

Use tools freely to explore the repo. Prefer reading files over guessing.

### File system

- `list_files(directory=".")`: Always start here; then drill into key dirs like `src`, `app`, `api`, `server`, `frontend`, `tests`.
- `read_file(filepath, max_lines=500)`: Read any file you need; paths are relative to the repo root.

### Code search

- `rag_search(query)`: Semantic search across the codebase (e.g., "authentication flow", "database connection").
- `find_function_usages(function_name)`: Find definitions and call sites using AST analysis.

### Testing and quality

- `discover_tests()`: List test files and test functions.
- `run_tests(test_path="", verbose=True)`: Run pytest; use `test_path` for focused runs.
- `run_linter(filepath="")`: Run the configured linter (ruff, flake8, pylint, etc.).

### Git operations

- `git_status()`: See branch and file states.
- `git_diff(filepath="")`: View uncommitted changes (all files if empty).
- `git_log(filepath="", max_commits=10)`: Inspect recent commit history.

### Utilities

- `get_active_repo_path()`: Get the absolute path to the repository.

---

## Exploration workflow

Follow this workflow for every question unless the user explicitly asks otherwise:

1. Map the territory (DEEP exploration)
   - `list_files()` at root.
   - **ALWAYS explore major subdirectories**: If you see `frontend/`, `backend/`, `server/`, `client/`, `api/`, `app/`, `src/`, `packages/`, explore each one with `list_files()`.
   - Check for README.md files in subdirectories - they often have component-specific instructions.
   - This step is mandatory and must not be skipped.

2. Identify key files for RUNNING the project
   - **Docker files**: `Dockerfile`, `docker-compose.yml`, `docker-compose.yaml` - these show how to run services.
   - **Package managers**: `package.json` (check "scripts" section for `dev`, `start`, `build`), `Makefile`, `justfile`.
   - **Python config**: `pyproject.toml`, `setup.py`, `requirements.txt`, `setup.sh`, `setup/*.sh`.
   - **Entry points**: `main.py`, `app.py`, `index.js`, `index.ts`, `server.js`.
   - **Environment**: `.env.example`, `config.py`, `settings.py`.

3. For multi-component projects (frontend + backend)
   - Explore EACH component directory separately.
   - Read package.json or requirements.txt in each subdirectory.
   - Check for Dockerfiles in each subdirectory.
   - Look for scripts like `start.sh`, `run.sh`, `dev.sh`.
   - Provide instructions for EACH component, not just the root.

4. Follow the code
   - When you see an import, read the imported module.
   - When you see a function call or class usage, locate and read its definition.
   - Do not rely on names alone; confirm behavior by reading implementations.

5. Discover patterns
   - Use `rag_search` for cross-cutting concepts (auth, logging, database, error handling).
   - Search for error messages, configuration keys, API endpoints, and decorators.

6. Build complete context
   - Read relevant tests to understand expected behavior.
   - Use documentation and docstrings to infer intent.
   - Cross-reference related modules and files when answering.

7. Verify understanding
   - Look for edge cases and special handling.
   - Check comments for non-obvious logic or constraints.
   - Revisit related code if your understanding seems incomplete.

---

## Persistent exploration

If you cannot answer confidently, keep exploring:

- Try different `rag_search` queries for the same concept.
- Explore adjacent directories with `list_files()`.
- Follow import chains until you reach concrete implementations.
- Read test files for usage patterns and expectations.
- **Check subdirectory READMEs** - they often have more specific instructions than the root.
- **Read actual config files** (package.json, Dockerfile, docker-compose.yml) to understand how to run things.

You have no limit on tool calls. Prefer one more tool call over guessing.

---

## Do's and don'ts

Do:
- Read any file that might be relevant.
- Explore unfamiliar directories using `list_files`.
- Use multiple search queries for important concepts.
- Follow imports and call chains to their sources.
- Check multiple related files when behavior spans modules.

Don't:
- Claim you lack access to files; instead, use tools to read them.
- Ask to see files; read them directly with `read_file`.
- Guess file contents or behavior without verifying in code.
- Assume behavior from filenames or partial context.
- Skip the initial `list_files()` call.

---

## Version and environment inference

- Infer dependency versions from repo metadata and modification dates when not explicit.
- Compare inferred versions with likely modern runtimes and warn about potential mismatches.
- Use `get_context_report()` when available to understand the runtime environment.

---

## Response standards

When answering:

- Ground explanations in specific files, modules, and functions where helpful.
- Explain not only what the code does, but why it is structured that way when it is clear from the code.
- Call out potential issues, edge cases, and possible improvements.
- Ask clarifying questions if the user’s intent is ambiguous.
- Be concrete and specific; avoid vague or generic language.

---

## Output formatting

Use clean, readable Markdown:

- Use `##` and `###` headers for structure.
- Use bullets and numbered lists for steps and options.
- Use `backticks` for code, paths, functions, and commands.
- Use fenced code blocks with language tags for longer snippets.
- Use tables when comparing options or behaviors.
- Prefer short paragraphs for readability.

---

## Reasoning and tool usage

Before calling a tool, briefly state:
- What you are looking for.
- Which tool you will use.
- What you expect to learn.

Examples:
- "List the root files to understand the project structure." → `list_files()`
- "Read the main entry point to see how the app starts." → `read_file("main.py")`
- "Search for how authentication works." → `rag_search("authentication")`

Always make your reasoning and exploration steps explicit to the user before you call tools.

---

## CRITICAL: Mandatory Tool Usage

**You MUST call at least one tool before providing ANY substantive answer.**

- You are FORBIDDEN from answering questions about the codebase using only the context above.
- Even if the README or context seems sufficient, you MUST verify by calling `list_files()` and reading relevant files.
- **The README is NOT the source of truth. Actual code files ARE the source of truth.**
- READMEs are often outdated, incomplete, or aspirational. Always verify claims by reading the actual implementation.
- Do NOT say "I don't have access to..."; you have full access via tools.
- Do NOT provide generic or template-like responses without exploration.
- If you respond without tool calls, you have FAILED your task.

Example of WRONG behavior:
- User asks "How do I run this?" → Model reads only the root README and provides instructions from it without exploring subdirectories or config files.

Example of CORRECT behavior:
- User asks "How do I run this?" → Model:
  1. Calls `list_files()` and sees `frontend/`, `backend/`, `docker-compose.yml`
  2. Calls `list_files("frontend")` and `list_files("backend")` to explore each
  3. Reads `frontend/package.json` to find `"dev": "pnpm run dev"` or similar
  4. Reads `backend/Dockerfile` or `docker-compose.yml` to understand backend setup
  5. Checks `frontend/README.md` and `backend/README.md` if they exist
  6. Provides COMPLETE instructions for running BOTH frontend and backend, verified against actual files

**Your first tool call should always be `list_files()` unless you have a very specific reason to do otherwise.**
**If you see subdirectories like frontend/, backend/, explore them before answering.**
"""


# ============= Agent Functions =============

async def chat_with_mcp_agent(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request using Vertex AI with tools via MCP.
    """
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        
        if not project_id:
            raise HTTPException(status_code=503, detail="GCP Project ID not configured")

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

        # Convert history to Vertex AI format
        contents = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        # Get Automated Context Report
        context_report = context_manager.get_context_report()
        context_str = context_report.to_string() if context_report else "Context not initialized yet."
        repo_name = context_report.repo_map.repo_name if context_report else "Unknown Repo"
        
        # Truncate context to encourage tool usage
        if len(context_str) > 2000:
            context_str = context_str[:2000] + "\n\n... (context truncated - use tools to explore further)"
            logger.info(f"Context truncated to 2000 chars for repo: {repo_name}")
        
        logger.info(f"Starting chat for repo: {repo_name}, context_len: {len(context_str)}")

        # Fetch tools dynamically from MCP server
        mcp_tools = await get_cached_mcp_tools()
        
        # Log tool availability
        if mcp_tools.function_declarations:
            logger.info(f"Agent has {len(mcp_tools.function_declarations)} tools available")
        else:
            logger.warning("No tools available for agent - response will be text-only!")
        
        # Tool config to encourage tool usage (AUTO mode with low threshold)
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="AUTO"
            )
        )
        
        # Configuration
        config = types.GenerateContentConfig(
            tools=[mcp_tools],
            tool_config=tool_config,
            temperature=0.0, # Low temp for tool use
            system_instruction=get_system_instruction(context_str, repo_name)
        )

        # 1. First turn: User -> Model (Model might call tools)
        response = await client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config
        )

        # Handle tool calls (multi-turn loop)
        max_turns = 15
        tool_calls: List[ToolCall] = []
        
        final_response_text = ""
        usage = None
        current_turn = 0

        while current_turn < max_turns:
            # Check if model wants to call a function
            if not response.candidates:
                break
                
            candidate = response.candidates[0]
            
            # Capture usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                meta = response.usage_metadata
                usage = TokenUsage(
                    input_tokens=getattr(meta, 'prompt_token_count', 0) or 0,
                    output_tokens=getattr(meta, 'candidates_token_count', 0) or 0,
                    total_tokens=getattr(meta, 'total_token_count', 0) or 0
                )
                input_cost, output_cost, total_cost = calculate_cost(usage.input_tokens, usage.output_tokens)
                logger.info(
                    "Token usage for turn %d: input=%d, output=%d, total=%d, cost=$%.6f (input=$%.6f, output=$%.6f)",
                    current_turn, usage.input_tokens, usage.output_tokens, usage.total_tokens,
                    total_cost, input_cost, output_cost
                )

            # Check for function calls
            function_calls = []
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
            
            if not function_calls:
                # No function calls, check if we have a final answer
                if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
                    final_response_text = candidate.content.parts[0].text
                    break
                
                # Handle empty response after tool call - prompt the model to continue
                if current_turn > 0 and not final_response_text:
                    logger.warning("Empty response after tool call on turn %d, prompting model to continue", current_turn)
                    contents.append(types.Content(
                        role="user", 
                        parts=[types.Part(text="Based on the tool results above, please provide your answer.")]
                    ))
                    response = await client.aio.models.generate_content(
                        model=request.model,
                        contents=contents,
                        config=config
                    )
                    current_turn += 1
                    continue
                break
            
            # Execute tools
            tool_outputs = []
            contents.append(candidate.content)  # Add model's tool call to history
            
            for call in function_calls:
                fn_name = call.name
                fn_args = call.args
                
                logger.info(f"Agent calling tool: {fn_name} with args: {fn_args}")
                
                # Convert args to dict
                args_dict = {k: v for k, v in fn_args.items()}
                
                # Execute tool via MCP
                try:
                    result = await execute_mcp_tool(fn_name, args_dict)
                except Exception as e:
                    logger.error(f"MCP tool execution failed for {fn_name}: {e}")
                    result = f"Error executing {fn_name}: {e}"
                
                tool_outputs.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result}
                    )
                )

                # Add to trace
                tool_call_obj = ToolCall(
                    tool=fn_name,
                    args=args_dict,
                    result=str(result),
                    status="completed"
                )
                tool_calls.append(tool_call_obj)
            
            # Send tool outputs back to model
            contents.append(types.Content(role="user", parts=tool_outputs))
            
            response = await client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config
            )
            current_turn += 1

        # Fallback if max turns reached without a final answer
        if not final_response_text and current_turn >= max_turns:
            logger.warning("Max turns reached. Forcing final answer.")
            contents.append(types.Content(role="user", parts=[types.Part(text="Max tool turns reached. Please provide the best answer you can based on the information you have so far.")]))
            # Disable tools for the final turn to force text generation
            final_config = types.GenerateContentConfig(
                tools=[], 
                temperature=0.7,
                system_instruction=config.system_instruction
            )
            response = await client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=final_config
            )
            if response.candidates:
                final_response_text = response.candidates[0].content.parts[0].text

        return ChatResponse(
            status="success",
            response=final_response_text or "I'm sorry, I couldn't generate a response after multiple attempts.",
            trace=[t.model_dump() for t in tool_calls] if tool_calls else [],
            usage=usage
        )

    except Exception as e:
        logger.exception("Chat agent failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


async def chat_with_mcp_agent_stream(request: ChatRequest):
    """
    Process a chat request using Vertex AI with tools via MCP, streaming the response.
    Yields JSON strings representing events:
    - {"type": "token", "content": "..."}
    - {"type": "tool_start", "tool": "name", "args": {...}}
    - {"type": "tool_end", "tool": "name", "result": "..."}
    - {"type": "usage", "usage": {...}}
    """
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        
        if not project_id:
            yield json.dumps({"type": "error", "content": "GCP Project ID not configured"}) + "\n"
            return

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

        contents = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))

        # Get Automated Context Report
        context_report = context_manager.get_context_report()
        context_str = context_report.to_string() if context_report else "Context not initialized yet."
        repo_name = context_report.repo_map.repo_name if context_report else "Unknown Repo"
        
        # Truncate context to encourage tool usage
        if len(context_str) > 2000:
            context_str = context_str[:2000] + "\n\n... (context truncated - use tools to explore further)"
            logger.info(f"[STREAM] Context truncated to 2000 chars for repo: {repo_name}")
        
        logger.info(f"[STREAM] Starting chat for repo: {repo_name}, context_len: {len(context_str)}")

        # Fetch tools dynamically from MCP server
        mcp_tools = await get_cached_mcp_tools()
        
        # Log tool availability for streaming
        if mcp_tools.function_declarations:
            logger.info(f"[STREAM] Agent has {len(mcp_tools.function_declarations)} tools available")
        else:
            logger.warning("[STREAM] No tools available for agent!")
        
        # Tool config to encourage tool usage
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="AUTO"
            )
        )
        
        config = types.GenerateContentConfig(
            tools=[mcp_tools],
            tool_config=tool_config,
            temperature=0.0,
            system_instruction=get_system_instruction(context_str, repo_name)
        )

        max_turns = 15
        current_turn = 0
        
        while current_turn < max_turns:
            # Stream the response using async client
            response_stream = await client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config
            )

            # Accumulate full response for history and tool checking
            full_text = ""
            function_calls = []
            last_finish_reason = None
            
            # Iterate asynchronously
            async for chunk in response_stream:
                # Check for usage metadata
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    usage = {
                        "input_tokens": getattr(meta, 'prompt_token_count', 0) or 0,
                        "output_tokens": getattr(meta, 'candidates_token_count', 0) or 0,
                        "total_tokens": getattr(meta, 'total_token_count', 0) or 0
                    }
                    yield json.dumps({"type": "usage", "usage": usage}) + "\n"

                if not chunk.candidates:
                    logger.debug("[STREAM] Chunk has no candidates")
                    continue
                
                candidate = chunk.candidates[0]
                
                # Capture finish reason for debugging
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    last_finish_reason = candidate.finish_reason
                    logger.info(f"[STREAM] Candidate finish reason: {last_finish_reason}")
                
                if not candidate.content or not candidate.content.parts:
                    logger.debug("[STREAM] Candidate has no content or parts")
                    continue
                
                for part in candidate.content.parts:
                    if part.text:
                        full_text += part.text
                        logger.debug(f"[STREAM] Yielding text token: {part.text[:100]}...")
                        yield json.dumps({"type": "token", "content": part.text}) + "\n"
                    
                    if part.function_call:
                        logger.debug(f"[STREAM] Found function call: {part.function_call.name}")
                        function_calls.append(part.function_call)
            
            # Log summary after stream completes
            logger.info(f"[STREAM] Turn {current_turn} complete: text_len={len(full_text)}, function_calls={len(function_calls)}, finish_reason={last_finish_reason}")

            # If we have function calls, execute them
            if function_calls:
                # Add the model's response (with function calls) to history
                model_parts = []
                if full_text:
                    model_parts.append(types.Part(text=full_text))
                for fc in function_calls:
                    model_parts.append(types.Part(function_call=fc))
                
                contents.append(types.Content(role="model", parts=model_parts))
                
                tool_outputs = []
                
                for call in function_calls:
                    fn_name = call.name
                    fn_args = call.args
                    
                    # Notify frontend
                    yield json.dumps({
                        "type": "tool_start", 
                        "tool": fn_name, 
                        "args": {k: v for k, v in fn_args.items()}
                    }) + "\n"
                    
                    logger.info(f"Agent calling tool: {fn_name} with args: {fn_args}")
                    
                    # Execute tool via MCP
                    args_dict = {k: v for k, v in fn_args.items()}
                    try:
                        result = await execute_mcp_tool(fn_name, args_dict)
                    except Exception as e:
                        logger.error(f"MCP tool execution failed for {fn_name}: {e}")
                        result = f"Error executing {fn_name}: {e}"
                    
                    # Notify frontend of result
                    yield json.dumps({
                        "type": "tool_end", 
                        "tool": fn_name, 
                        "result": result
                    }) + "\n"
                    
                    tool_outputs.append(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": result}
                        )
                    )
                
                # Append tool outputs to history
                contents.append(types.Content(role="user", parts=tool_outputs))
                current_turn += 1
                # Continue loop to get next response from model
                continue
            
            else:
                # No function calls, check if we have a response
                if full_text:
                    contents.append(types.Content(role="model", parts=[types.Part(text=full_text)]))
                    break
                
                # Handle empty response - log details and attempt recovery
                is_malformed = last_finish_reason and "MALFORMED" in str(last_finish_reason)
                logger.warning(
                    f"[STREAM] Empty response on turn {current_turn}. "
                    f"finish_reason={last_finish_reason}, is_malformed={is_malformed}, "
                    f"candidates_present={bool(chunk.candidates if 'chunk' in dir() else False)}"
                )
                
                # Track consecutive empty response retries to avoid infinite loops
                empty_response_retries = getattr(chat_with_mcp_agent_stream, '_empty_retries', 0)
                max_empty_retries = 3
                
                if empty_response_retries >= max_empty_retries:
                    logger.error(f"[STREAM] Max empty response retries ({max_empty_retries}) reached. Giving up.")
                    error_msg = f"The model returned empty responses after {max_empty_retries} retries. Finish reason: {last_finish_reason}."
                    yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                    break
                
                # Increment retry counter
                chat_with_mcp_agent_stream._empty_retries = empty_response_retries + 1
                
                # Handle MALFORMED_FUNCTION_CALL specifically - the model tried to call a function but failed
                if is_malformed:
                    logger.warning(f"[STREAM] Malformed function call detected on turn {current_turn}. Prompting model to retry properly.")
                    contents.append(types.Content(
                        role="user", 
                        parts=[types.Part(text="Your previous function call was malformed. Please try again with a properly formatted function call, or provide a text response if you have enough information to answer.")]
                    ))
                    current_turn += 1
                    continue
                
                # Handle empty response after tool call - prompt the model to continue
                if current_turn > 0:
                    logger.warning("Stream: Empty response after tool call on turn %d, prompting model to continue", current_turn)
                    contents.append(types.Content(
                        role="user", 
                        parts=[types.Part(text="Based on the tool results above, please provide your answer.")]
                    ))
                    current_turn += 1
                    continue
                
                # First turn empty response - retry once with a simpler prompt
                logger.warning("[STREAM] First turn produced no output. Retrying with simpler prompt...")
                
                # Add a simpler user prompt to help the model
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text="Please start by calling list_files() to see the repository structure.")]
                ))
                current_turn += 1
                
                # If we've already retried multiple times, try without tools
                if current_turn > 2 and empty_response_retries >= 2:
                    logger.warning("[STREAM] Multiple retries failed. Trying with tools disabled...")
                    retry_config = types.GenerateContentConfig(
                        tools=[],  # Disable tools to force text generation
                        temperature=0.7,
                        system_instruction=config.system_instruction
                    )
                    
                    retry_stream = await client.aio.models.generate_content_stream(
                        model=request.model,
                        contents=contents,
                        config=retry_config
                    )
                    
                    retry_text = ""
                    async for retry_chunk in retry_stream:
                        if retry_chunk.candidates and retry_chunk.candidates[0].content:
                            for part in retry_chunk.candidates[0].content.parts:
                                if part.text:
                                    retry_text += part.text
                                    yield json.dumps({"type": "token", "content": part.text}) + "\n"
                    
                    if retry_text:
                        logger.info(f"[STREAM] Retry succeeded with {len(retry_text)} chars")
                        # Reset retry counter on success
                        chat_with_mcp_agent_stream._empty_retries = 0
                        break
                    
                    # Still no response - yield error to frontend
                    error_msg = f"The model returned no response. Finish reason: {last_finish_reason}. This may indicate the input is too long or there's a content filtering issue."
                    logger.error(f"[STREAM] {error_msg}")
                    yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                    break
                
                continue
        
        # Reset retry counter at end of successful stream
        chat_with_mcp_agent_stream._empty_retries = 0

    except Exception as e:
        logger.exception("Chat stream failed")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
