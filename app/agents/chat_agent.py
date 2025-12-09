"""
Chat Agent
Handles user queries using Vertex AI with function calling (tools).
Tools:
- RAG (query_db)
- Filesystem (list_files, read_file)
- Dependencies (list_package_files)
"""
import logging
import os
import json
from typing import List, Dict, Any

from fastapi import HTTPException
from google import genai
from google.genai import types

from ..models import ChatRequest, ChatResponse, TokenUsage, ToolCall
from ..tools.filesystem import read_file, list_files
from ..tools.shell_execution import execute_command as shell_execute, ensure_venv
from ..tools.test_runner import discover_tests as test_discover, run_tests as test_run
from ..tools.git_tools import get_git_status, get_git_diff, get_git_log
from ..tools.code_quality import run_linter as quality_lint
from ..tools.ast_tools import find_function_usages as ast_find_usages
from ..search import perform_hybrid_search
from ..active_repo_state import active_repo_state

# MCP Client imports
try:
    from mcp import ClientSession, types as mcp_types
    from mcp.client.streamable_http import streamablehttp_client
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pricing for gemini-2.5-flash (per million tokens)
INPUT_PRICE_PER_MILLION = 0.30  # $0.30 per million tokens
OUTPUT_PRICE_PER_MILLION = 2.50  # $2.50 per million tokens

def calculate_cost(input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
    """Calculate cost based on token usage.
    Returns: (input_cost, output_cost, total_cost) in USD"""
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost

from ..repo_manager import repo_manager

# --- Tool Wrappers for Vertex AI ---

def get_active_repo_path() -> str:
    """Helper to get the path of the active repository."""
    repo_id = active_repo_state.get_active_repo_id()
    if repo_id and repo_id != "default":
        return os.path.join(repo_manager.repos_base_path, repo_id)
    return "."



def read_file_wrapper(filepath: str, max_lines: int = 500) -> str:
    """
    Read the content of a file.
    Args:
        filepath: Path to the file.
        max_lines: Maximum number of lines to read.
    """
    try:
        base_path = get_active_repo_path()
        full_path = os.path.join(base_path, filepath)
        return read_file(full_path, max_lines)
    except Exception as e:
        return f"Error reading file: {e}"

def rag_search(query: str) -> str:
    """
    Search the codebase using the RAG system (Vector + Keyword search).
    Use this to find relevant code snippets or answer questions about the codebase logic.
    """
    try:
        # Get active repository ID for filtering
        repo_id = active_repo_state.get_active_repo_id()
        results = perform_hybrid_search(query, n_results=5, repo_id=repo_id)
        if not results:
            return "No relevant code found."
        
        # Format results for the LLM
        context = []
        for r in results:
            meta = r['metadata']
            context.append(f"File: {meta.get('filepath')}\nLines: {meta.get('start_line')}-{meta.get('end_line')}\nContent:\n{r['content']}\n")
            
        return "\n---\n".join(context)
    except Exception as e:
        return f"RAG search failed: {e}"

def list_files_wrapper(directory: str = ".") -> str:
    """
    List files and subdirectories in a directory.
    Args:
        directory: Relative path within the repo (default is root).
    """
    try:
        base_path = get_active_repo_path()
        full_path = os.path.join(base_path, directory)
        return list_files(full_path)
    except Exception as e:
        return f"Error listing files: {e}"


# --- New Tool Wrappers ---

def execute_command_wrapper(command: str, timeout: int = 60) -> str:
    """Execute a shell command in the repository's virtual environment."""
    try:
        repo_path = get_active_repo_path()
        result = shell_execute(command, repo_path=repo_path, timeout=timeout, use_venv=True)
        
        output = f"Command: {result.get('command_executed', command)}\n"
        output += f"Exit Code: {result.get('exit_code', -1)}\n"
        if result.get('venv_used'):
            output += "(Executed in virtual environment)\n"
        output += "\n--- STDOUT ---\n" + result.get('stdout', '')
        if result.get('stderr'):
            output += "\n--- STDERR ---\n" + result.get('stderr', '')
        if result.get('timed_out'):
            output += f"\n[TIMED OUT after {timeout}s]"
        return output
    except Exception as e:
        return f"Error executing command: {e}"


def discover_tests_wrapper() -> str:
    """Discover all pytest tests in the repository."""
    try:
        repo_path = get_active_repo_path()
        result = test_discover(repo_path)
        
        output = f"Test Discovery Results:\n"
        output += f"- Test files: {len(result.get('test_files', []))}\n"
        output += f"- Test functions: {result.get('test_count', 0)}\n\n"
        
        if result.get('test_files'):
            output += "Test Files:\n"
            for f in result['test_files'][:20]:  # Limit to 20
                output += f"  - {f}\n"
            if len(result['test_files']) > 20:
                output += f"  ... and {len(result['test_files']) - 20} more\n"
        
        return output
    except Exception as e:
        return f"Error discovering tests: {e}"


def run_tests_wrapper(test_path: str = "", verbose: bool = True) -> str:
    """Run pytest tests in the repository."""
    try:
        repo_path = get_active_repo_path()
        result = test_run(repo_path, test_path=test_path, verbose=verbose)
        
        output = f"Test Results:\n"
        output += f"- Passed: {result.get('passed', 0)}\n"
        output += f"- Failed: {result.get('failed', 0)}\n"
        output += f"- Errors: {result.get('errors', 0)}\n"
        output += f"- Skipped: {result.get('skipped', 0)}\n"
        output += f"- Duration: {result.get('duration_seconds', 0):.2f}s\n\n"
        output += "--- Output ---\n" + result.get('output', '')
        return output
    except Exception as e:
        return f"Error running tests: {e}"


def git_status_wrapper() -> str:
    """Get the git status of the repository."""
    try:
        repo_path = get_active_repo_path()
        result = get_git_status(repo_path)
        
        if not result.get('success'):
            return f"Git status failed: {result.get('output', 'Unknown error')}"
        
        output = f"Branch: {result.get('branch', 'unknown')}\n"
        output += f"Status: {'Clean' if result.get('clean') else 'Modified'}\n"
        
        if result.get('modified'):
            output += f"\nModified files ({len(result['modified'])}):\n"
            for f in result['modified'][:10]:
                output += f"  M {f}\n"
        
        if result.get('staged'):
            output += f"\nStaged files ({len(result['staged'])}):\n"
            for f in result['staged'][:10]:
                output += f"  + {f}\n"
        
        if result.get('untracked'):
            output += f"\nUntracked files ({len(result['untracked'])}):\n"
            for f in result['untracked'][:10]:
                output += f"  ? {f}\n"
        
        return output
    except Exception as e:
        return f"Error getting git status: {e}"


def git_diff_wrapper(filepath: str = "") -> str:
    """Get git diff for the repository or a specific file."""
    try:
        repo_path = get_active_repo_path()
        result = get_git_diff(repo_path, filepath=filepath)
        
        if not result.get('success') and not result.get('diff'):
            return "No changes to show."
        
        output = f"Files changed: {result.get('files_changed', 0)}\n"
        output += f"Insertions: +{result.get('insertions', 0)}\n"
        output += f"Deletions: -{result.get('deletions', 0)}\n\n"
        output += result.get('diff', '')
        return output
    except Exception as e:
        return f"Error getting git diff: {e}"


def git_log_wrapper(filepath: str = "", max_commits: int = 10) -> str:
    """Get git log for the repository or a specific file."""
    try:
        repo_path = get_active_repo_path()
        result = get_git_log(repo_path, filepath=filepath, max_commits=max_commits)
        
        if not result.get('success'):
            return f"Git log failed: {result.get('output', 'Unknown error')}"
        
        output = f"Recent commits ({len(result.get('commits', []))}):\n\n"
        for commit in result.get('commits', []):
            output += f"{commit.get('short_hash', '')} - {commit.get('message', '')}\n"
            output += f"  Author: {commit.get('author', '')} | {commit.get('date', '')}\n\n"
        
        return output
    except Exception as e:
        return f"Error getting git log: {e}"


def run_linter_wrapper(filepath: str = "") -> str:
    """Run linter on the repository or a specific file."""
    try:
        repo_path = get_active_repo_path()
        result = quality_lint(repo_path, filepath=filepath)
        
        output = f"Linter: {result.get('linter_used', 'unknown')}\n"
        output += f"Issues found: {result.get('issue_count', 0)}\n\n"
        
        if result.get('issues'):
            for issue in result['issues'][:20]:  # Limit to 20
                output += f"{issue.get('file')}:{issue.get('line')}: {issue.get('message')}\n"
            if len(result['issues']) > 20:
                output += f"\n... and {len(result['issues']) - 20} more issues\n"
        
        return output
    except Exception as e:
        return f"Error running linter: {e}"


def find_function_usages_wrapper(function_name: str) -> str:
    """Find all usages of a function in the codebase."""
    try:
        repo_path = get_active_repo_path()
        result = ast_find_usages(repo_path, function_name, include_definitions=True)
        
        output = f"Searching for usages of '{function_name}':\n\n"
        
        if result.get('definitions'):
            output += f"Definitions ({len(result['definitions'])}):\n"
            for defn in result['definitions']:
                output += f"  {defn.get('file')}:{defn.get('line')}\n"
            output += "\n"
        
        output += f"Usages ({result.get('usage_count', 0)}):\n"
        for usage in result.get('usages', [])[:30]:  # Limit to 30
            output += f"  {usage.get('file')}:{usage.get('line')} ({usage.get('type')})\n"
        
        if result.get('usage_count', 0) > 30:
            output += f"\n... and {result['usage_count'] - 30} more usages\n"
        
        return output
    except Exception as e:
        return f"Error finding function usages: {e}"


# Tools are now provided exclusively by the MCP server
# See mcp_server.py for available tools:
# - rag_search, read_file, list_files, execute_command
# - discover_tests, run_tests, git_status, git_diff, git_log
# - run_linter, find_function_usages, get_active_repo_path



# Dynamic tool schema loading from MCP server
# No static tool definitions - tools are fetched from MCP at runtime

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
                
                logger.info(f"Loaded {len(function_declarations)} tools from MCP server")
                return types.Tool(function_declarations=function_declarations)
                
    except Exception as e:
        logger.error(f"Failed to fetch MCP tools: {e}")
        return types.Tool(function_declarations=[])

# Cache for MCP tools to avoid fetching on every request
_cached_mcp_tools: types.Tool = None
_mcp_tools_cache_time: float = 0
MCP_TOOLS_CACHE_TTL = 60  # Cache tools for 60 seconds

async def get_cached_mcp_tools() -> types.Tool:
    """Get MCP tools with caching."""
    global _cached_mcp_tools, _mcp_tools_cache_time
    import time
    
    now = time.time()
    if _cached_mcp_tools is None or (now - _mcp_tools_cache_time) > MCP_TOOLS_CACHE_TTL:
        _cached_mcp_tools = await get_mcp_tools_as_vertex()
        _mcp_tools_cache_time = now
    
    return _cached_mcp_tools



from ..context_manager import context_manager
from datetime import datetime

def get_readme_content() -> str:
    """Read README.md from the active repository root if it exists."""
    try:
        repo_path = get_active_repo_path()
        readme_path = os.path.join(repo_path, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Limit to first 3000 chars to avoid context overflow
                if len(content) > 3000:
                    content = content[:3000] + "\n\n... (README truncated)"
                return content
    except Exception as e:
        logger.warning(f"Failed to read README.md: {e}")
    return ""

def get_system_instruction(context_str: str, repo_name: str) -> str:
    """Generate the system instruction with context and CoT protocol."""
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # Read README.md if available
    readme_content = get_readme_content()
    readme_section = ""
    if readme_content:
        readme_section = f"""
=== PROJECT README ===
{readme_content}
======================
"""
    
    return f"""You are an expert software developer and coding assistant for the '{repo_name}' repository.
You have full access to the source code of '{repo_name}' and are answering questions specifically about it.

**Current Date:** {current_date}
{readme_section}
=== AUTOMATED CONTEXT ===
{context_str}
=========================

=== YOUR TOOLS (USE THEM LIBERALLY) ===

You have access to these tools to explore and understand the codebase. Use them extensively!

**1. list_files(directory=".")**
   - Lists all files and subdirectories in a directory
   - ALWAYS call this first to see the full repo structure
   - Then drill into subdirectories: `list_files("src")`, `list_files("tests")`, etc.
   - Use this whenever you encounter an unfamiliar directory


**2. read_file(filepath, max_lines=500)**
   - Reads the full content of any file
   - Use this to examine source code, configs, README, requirements, etc.
   - Path is relative to repo root (e.g., "src/main.py")
   - USE THIS LIBERALLY - read every file you're curious about
   - If a file is truncated at 500 lines, read it in chunks or call again with a different range


**3. rag_search(query)**
   - Semantic search across the entire codebase
   - Best for finding: function implementations, class definitions, usage patterns, error handling
   - Example queries: "how is authentication handled", "database connection", "error handling", "API endpoints"
   - Great for discovering patterns without reading every file


**4. get_context_report()**
   - Returns environment info (OS, Python version), repo structure, and dependencies
   - Already provided above, but call this if you need a refresh or updated information


**5. get_active_repo_path()**
   - Returns the absolute filesystem path to the repository
   - Useful if you need to understand full file system paths


**6. execute_command(command, timeout=60)**
   - Execute shell commands in the repo's virtual environment (auto-creates .venv if missing)
   - Allowlisted commands only: pytest, pip list/install, git status/diff/log, ruff/flake8/mypy
   - Example: `execute_command("pip list")`, `execute_command("pytest tests/")`


**7. discover_tests()**
   - Find all pytest test files and functions in the repository
   - Use this before running tests to understand what's available


**8. run_tests(test_path="", verbose=True)**
   - Run pytest tests in the repository
   - Can run all tests or specific file: `run_tests("tests/test_api.py")`


**9. git_status()**
   - Get current branch, modified/staged/untracked files
   - Use this to understand the current state of changes


**10. git_diff(filepath="")**
   - View code changes (working directory vs HEAD)
   - Can view all changes or specific file


**11. git_log(filepath="", max_commits=10)**
   - View recent commit history
   - Useful for understanding when/why code changed


**12. run_linter(filepath="")**
   - Run code linter (auto-detects ruff/flake8/pylint)
   - Reports style issues and potential bugs


**13. find_function_usages(function_name)**
   - Find all places a function is defined and called
   - Uses AST analysis for accurate results


=== EXPLORATION WORKFLOW (CRITICAL) ===

Follow this EXACT workflow when answering any question:

**Step 1: FIRST CALL - Directory Structure**
   - Call `list_files()` to see all files in repository root
   - This is MANDATORY - do not skip
   - Then call `list_files("subdir")` for subdirectories you're interested in


**Step 2: Identify Entry Points & Key Files**
   - main.py, app.py, index.js, __init__.py
   - README.md for project overview and setup instructions
   - config.py, settings.py, .env.example for configuration
   - requirements.txt, pyproject.toml, package.json for dependencies


**Step 3: Systematic Exploration**
   - When you see imports, read those files to understand dependencies
   - When you find a function call, search for its definition
   - When you see a class, read its full implementation
   - Follow the import chain completely - don't assume based on naming


**Step 4: Pattern Discovery**
   - Use `rag_search` to find all usages of specific functions/classes
   - Search for error messages and exception handling
   - Look for configuration keys and API endpoints
   - Search for decorators and special patterns


**Step 5: Comprehensive Context Building**
   - Read tests to understand expected behavior (often more reliable than comments)
   - Read documentation and docstrings for intent and usage
   - Read config files to understand available options
   - Cross-reference related files


**Step 6: Verify Understanding**
   - Once you have information, search for edge cases or alternative implementations
   - Look for comments explaining non-obvious logic
   - Check for error handling and validation


=== AGGRESSIVE EXPLORATION PRINCIPLE ===

**DO NOT GIVE UP EARLY.** If you don't find what you need:

- **Try different search terms** - Rephrase and use `rag_search` with different keywords
- **Explore adjacent directories** - Keep calling `list_files()` on different folders
- **Read more files** - The answer might be in a related file you haven't checked yet
- **Follow import chains** - Read every imported module
- **Check test files** - Tests often reveal implementation details better than source code
- **Look for patterns** - Search for similar patterns using RAG

You have unlimited tool calls. Exploration is free. Keep going until you're confident.


=== CRITICAL RULES ===

**DO:**
- Call `read_file` on EVERY file you're curious about - it's free
- Use `list_files` to explore unfamiliar directories
- Use `rag_search` to find function/class usages and patterns
- Read MORE files rather than fewer - each file may contain crucial context
- Follow import chains all the way through
- Check multiple related files to build complete understanding
- Search with different query variations for comprehensive coverage


**DON'T:**
- NEVER respond with "I don't have access to..." - You DO have access, use your tools!
- NEVER say "I would need to see..." - Just read the file with your tools!
- NEVER guess what a file contains - Always read it first
- NEVER assume anything based on filename alone - Verify by reading
- NEVER give incomplete answers when exploring would only take one more tool call
- NEVER skip the initial `list_files()` step


=== SELF-VERIFICATION CHECKLIST ===

Before writing your final answer, ask yourself:

1. "Have I actually seen the relevant code, or am I making assumptions?"
2. "Are there other files in the codebase I should check?"
3. "Did I follow all import chains to their source?"
4. "Could reading one more file significantly improve my answer?"
5. "Have I searched for related patterns using RAG?"
6. "Are there edge cases I should look for?"


If you answer YES to any of these → investigate further first!


=== VERSION & DEPENDENCY INFERENCE ===

- If dependencies lack explicit versions, check the "Last Updated" date and infer versions accordingly
- Cross-reference with LTS versions and stability guidelines
- WARN the user if there's a significant time gap between repo age and their current environment packages
- Use `get_context_report()` to understand the environment and identify version conflicts


=== RESPONSE QUALITY STANDARDS ===

- Provide complete, accurate answers backed by the actual code you've read
- Reference specific files and line numbers when helpful
- Explain the "why" behind implementations, not just the "what"
- Identify potential issues or improvements you notice
- Suggest follow-up questions if the user's intent is unclear
- Be specific and concrete - avoid vague generalizations


=== OUTPUT FORMATTING (CRITICAL) ===

**ALWAYS format your responses in clean, readable Markdown:**

- Use **headers** (`##`, `###`) to organize sections
- Use **bullet points** and **numbered lists** for clarity
- Use **bold** and *italics* for emphasis
- Wrap ALL code, file paths, function names, and commands in backticks: `example`
- Use fenced code blocks with language tags for multi-line code:

```python
def example():
    return "Use this format for code"
```

- Use tables when comparing multiple items
- Keep paragraphs short and scannable


=== THINKING OUT LOUD (CRITICAL) ===

**ALWAYS explain your reasoning before calling a tool.** Before each tool call, briefly state:
- What you're looking for
- Why you're using that specific tool
- What you expect to find

For example:
- "Let me first list the files to understand the repository structure..." → then call list_files()
- "I'll read the main.py file to see the entry point..." → then call read_file("main.py")
- "Now I need to search for how authentication is implemented..." → then call rag_search("authentication")

This helps the user follow your reasoning process. Do NOT silently call tools - always provide context first.
"""

def get_system_instruction_wrapper(context_details: str = "") -> str:
    """Wrapper for get_system_instruction to be used as a tool."""
    return get_system_instruction(context_details)

def get_context_report_wrapper() -> str:
    """Wrapper for get_context_report to be used as a tool."""
    report = context_manager.get_context_report()
    return report.to_string() if report else "Context not initialized."




# MCP URL for local server
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")


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


async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request using Vertex AI with tools.
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

        # Fetch tools dynamically from MCP server
        mcp_tools = await get_cached_mcp_tools()
        
        # Configuration
        config = types.GenerateContentConfig(
            tools=[mcp_tools],
            temperature=0.0, # Low temp for tool use
            system_instruction=get_system_instruction(context_str, repo_name)
        )

        # 1. First turn: User -> Model (Model might call tools)
        # Use async client
        response = await client.aio.models.generate_content(
            model=request.model,
            contents=contents,
            config=config
        )

        # Handle tool calls (multi-turn loop)
        # We'll allow a max of 15 tool turns to prevent infinite loops
        max_turns = 15
        # Capture tool calls for trace
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
            contents.append(candidate.content) # Add model's tool call to history
            
            for call in function_calls:
                fn_name = call.name
                fn_args = call.args
                
                logger.info(f"Agent calling tool: {fn_name} with args: {fn_args}")
                
                # Convert args to dict
                args_dict = {k: v for k, v in fn_args.items()}
                
                # Execute tool via MCP (no legacy fallback)
                try:
                    if MCP_CLIENT_AVAILABLE:
                        result = await execute_mcp_tool(fn_name, args_dict)
                    else:
                        result = f"Error: MCP client not available, cannot execute tool {fn_name}"
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
                    args=args_dict if 'args_dict' in locals() else {k:v for k,v in fn_args.items()},
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


async def chat_with_agent_stream(request: ChatRequest):
    """
    Process a chat request using Vertex AI with tools, streaming the response.
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

        # Fetch tools dynamically from MCP server
        mcp_tools = await get_cached_mcp_tools()
        
        config = types.GenerateContentConfig(
            tools=[mcp_tools],
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
                    
                    # Execute tool via MCP (no legacy fallback)
                    args_dict = {k: v for k, v in fn_args.items()}
                    try:
                        if MCP_CLIENT_AVAILABLE:
                            result = await execute_mcp_tool(fn_name, args_dict)
                        else:
                            result = f"Error: MCP client not available, cannot execute tool {fn_name}"
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
                logger.warning(
                    f"[STREAM] Empty response on turn {current_turn}. "
                    f"finish_reason={last_finish_reason}, candidates_present={bool(chunk.candidates if 'chunk' in dir() else False)}"
                )
                
                # Handle empty response after tool call - prompt the model to continue
                if current_turn > 0:
                    logger.warning("Stream: Empty response after tool call on turn %d, prompting model to continue", current_turn)
                    contents.append(types.Content(
                        role="user", 
                        parts=[types.Part(text="Based on the tool results above, please provide your answer.")]
                    ))
                    current_turn += 1
                    continue
                
                # First turn empty response - retry once with tools disabled
                logger.warning("[STREAM] First turn produced no output. Retrying with tools disabled...")
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
                    break
                
                # Still no response - yield error to frontend
                error_msg = f"The model returned no response. Finish reason: {last_finish_reason}. This may indicate the input is too long or there's a content filtering issue."
                logger.error(f"[STREAM] {error_msg}")
                yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                break

    except Exception as e:
        logger.exception("Chat stream failed")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
