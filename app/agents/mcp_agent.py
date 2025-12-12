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

from ..core.models import ChatRequest, ChatResponse, TokenUsage, ToolCall
from ..repos.state import active_repo_state
from ..repos.manager import repo_manager
from ..repos.context import context_manager

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

# Pricing for gemini-3-pro-preview (per million tokens)
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

    return f"""You are an expert software developer and coding assistant for the '{repo_name}' repository.
You have full access to the source code of '{repo_name}' and are answering questions specifically about it.
Your goal is to help the user with their question by using the tools provided to you. Answer all the key points while keeping it concise.
AS SOON AS YOU HAVE SUFFICIENT INFORMATION, START ANSWERING THE USER'S QUESTION.

Below are useful tools and instructions to help you answer the user's question:

**Current Date:** {current_date}
**Note:** The "Last Updated" date provided in the context is approximate and inferred from the file system or git logs.

The readme is displayed here if it exists, you do not need to use tools to explore it again:
{readme_section}

=== AUTOMATED CONTEXT ===
{context_str}

Use your tools (`list_files` root, read configs) to classify the repo type *before* diving into answering the question (application or library).

=== AGGRESSIVE EXPLORATION PRINCIPLE ===
**DO NOT GIVE UP EARLY.** If you don't find what you need:

- **Try different search terms** - Rephrase and use `rag_search` with different keywords
- **Explore adjacent directories** - Keep calling `list_files()` on different folders
- **Read more files** - The answer might be in a related file you haven't checked yet
- **Follow import chains** - Read every imported module
- **Check test files** - Tests often reveal implementation details better than source code
- **Look for patterns** - Search for similar patterns using RAG

You have unlimited tool calls. Exploration is free. Keep going until you're confident.

=== RULES ===

**DO:**
- Always think and don't just copy answers from files. Sometimes answers are spread across multiple files.
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

=== SELF-VERIFICATION CHECKLIST ===

Before writing your final answer, ask yourself:

1. "Are there other files in the codebase I should check?"
2. "Did I follow all import chains to their source?"
3. "Are there edge cases I should look for?"
If you answer YES to any of these → investigate further first!


=== VERSION & DEPENDENCY INFERENCE ===

- If dependencies lack explicit versions, check the "Last Updated" date and infer versions accordingly
- Cross-reference with LTS versions and stability guidelines
- WARN the user if there's a significant time gap between repo age and their current environment packages
- Use `get_context_report()` to understand the environment and identify version conflicts


=== RESPONSE QUALITY STANDARDS ===

- Provide complete, accurate answers backed by the actual code you've read (but you can use your thoughts to help you answer)
- Identify potential issues or improvements you notice
- Suggest follow-up questions if the user's intent is unclear
- Be specific and concrete - avoid vague generalizations
- **CONCISENESS**: Keep answers brief and to the point. Do not ramble.
- **EFFICIENCY**: Do not repeat the same tool calls or process steps unnecessarily. If you have the info, use it.


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
- Be brief and to the point.

For example:
- "Let me list the files to understand the repository structure..." → then call list_files()
- "I'll read the main.py file to see the entry point..." → then call read_file("main.py")
- "Now I need to search for how authentication is implemented..." → then call rag_search("authentication")

This helps the user follow your reasoning process. Do NOT silently call tools - always provide context first.

=== SUGGESTIONS ===

- If the question is about how to run the repository, follow the steps in the README.md file or as follows:
    - Suggest cloning the github repo or using pip install depending on the nature of the repository.
    - Setting up a virtual environment.
    - Installing dependencies, noting the problems with dependency versions.
    - Anything else unique to the repository.


=== MOST IMPORTANT ===
**ALWAYS** make sure you provide a full final answer at the end of your response that directly addresses the user's question.
"""


# ============= RAG Context Injection =============

def get_rag_context_for_query(query: str, repo_id: Optional[str] = None) -> str:
    """
    Run a RAG query and format results as context for the model.
    Returns up to 5 chunks with their metadata, formatted as a hint.
    
    Args:
        query: The user's question to search for
        repo_id: Optional repository ID to filter results
        
    Returns:
        Formatted string with RAG results, or empty string if none found
    """
    try:
        from ..rag.search import perform_hybrid_search
        results = perform_hybrid_search(query, n_results=20, repo_id=repo_id)
        
        if not results:
            logger.info(f"No RAG results found for query: {query[:50]}...")
            return ""
        
        logger.info(f"Found {len(results)} RAG chunks for query: {query[:50]}...")
        
        context_parts = [
            "## RAG Search Results (Optional Starting Point)",
            "",
            "The following code snippets may be relevant to your question. Use these as hints, but always verify with tools:",
            ""
        ]
        
        for i, chunk in enumerate(results, 1):
            meta = chunk.get("metadata", {})
            content = chunk.get("content", "")
            # Truncate content to avoid overly long context
            if len(content) > 800:
                content = content[:800] + "\n... (truncated)"
            
            context_parts.append(f"### Chunk {i}")
            context_parts.append(f"- **File:** `{meta.get('filepath', 'unknown')}`")
            context_parts.append(f"- **Lines:** {meta.get('start_line', '?')}-{meta.get('end_line', '?')}")
            context_parts.append(f"- **Type:** {meta.get('chunk_type', 'unknown')}")
            context_parts.append(f"- **Relevance Score:** {chunk.get('combined_score', 0):.3f}")
            context_parts.append(f"```")
            context_parts.append(content)
            context_parts.append("```")
            context_parts.append("")
        
        return "\n".join(context_parts)
    except Exception as e:
        logger.warning(f"Failed to get RAG context: {e}")
        return ""


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
        
        # Get the last user message for RAG query
        last_user_message = None
        last_user_idx = -1
        for i, msg in enumerate(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                last_user_idx = i
        
        # Inject RAG context into the last user message
        repo_id = active_repo_state.get_active_repo_id()
        if last_user_message and last_user_idx >= 0:
            rag_context = get_rag_context_for_query(last_user_message, repo_id)
            if rag_context:
                # Create enhanced message with RAG context prepended
                enhanced_content = f"Use this content to jump where you need in your reasoning process:\n\n---RAG CONTENT---\n{rag_context}\n\n---\n\n**User Question:**\n{last_user_message}"
                # Update the message content for the model
                request.messages[last_user_idx].content = enhanced_content
                logger.info(f"Injected RAG context ({len(rag_context)} chars) into user message")
        
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
        
        # Get the last user message for RAG query
        last_user_message = None
        last_user_idx = -1
        for i, msg in enumerate(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                last_user_idx = i
        
        # Inject RAG context into the last user message
        repo_id = active_repo_state.get_active_repo_id()
        if last_user_message and last_user_idx >= 0:
            rag_context = get_rag_context_for_query(last_user_message, repo_id)
            if rag_context:
                # Create enhanced message with RAG context prepended
                enhanced_content = f"{rag_context}\n\n---\n\n**User Question:**\n{last_user_message}"
                # Update the message content for the model
                request.messages[last_user_idx].content = enhanced_content
                logger.info(f"[STREAM] Injected RAG context ({len(rag_context)} chars) into user message")
        
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
        # Local retry counters (avoid static function attributes for concurrent requests)
        malformed_retries = 0
        max_malformed_retries = 3
        empty_response_retries = 0
        max_empty_retries = 3
        
        while current_turn < max_turns:
            # Stream the response using async client
            response_stream = await client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config
            )

            # Accumulate full response for history and tool checking
            full_text = ""
            # Store original Part objects to preserve thought_signature for Gemini 3 Pro
            collected_parts = []  # List of original Part objects from the response
            function_call_parts = []  # Original parts containing function calls (preserves thought_signature)
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
                    # Collect all parts to preserve the original structure including thought_signature
                    collected_parts.append(part)
                    
                    if part.text:
                        full_text += part.text
                        logger.debug(f"[STREAM] Yielding text token: {part.text[:100]}...")
                        yield json.dumps({"type": "token", "content": part.text}) + "\n"
                    
                    if part.function_call:
                        logger.debug(f"[STREAM] Found function call: {part.function_call.name}")
                        # Store the ORIGINAL Part object, not just the function_call
                        # This preserves the thought_signature required by Gemini 3 Pro
                        function_call_parts.append(part)
            
            # Log summary after stream completes
            logger.info(f"[STREAM] Turn {current_turn} complete: text_len={len(full_text)}, function_calls={len(function_call_parts)}, finish_reason={last_finish_reason}")

            # If we have function calls, execute them
            if function_call_parts:
                # Add the model's response to history - use collected_parts directly
                # This preserves thought_signature which is REQUIRED by Gemini 3 Pro
                # If we have text + function calls, we need to build parts properly
                model_parts = []
                if full_text:
                    # Add text part if we accumulated text
                    model_parts.append(types.Part(text=full_text))
                # Add ORIGINAL function call parts (preserves thought_signature)
                model_parts.extend(function_call_parts)
                
                contents.append(types.Content(role="model", parts=model_parts))
                
                tool_outputs = []
                
                # Iterate over the original parts to execute function calls
                for part in function_call_parts:
                    call = part.function_call
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
                # Reset malformed retry counter on successful tool execution
                malformed_retries = 0
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
                if empty_response_retries >= max_empty_retries:
                    logger.error(f"[STREAM] Max empty response retries ({max_empty_retries}) reached. Giving up.")
                    error_msg = f"The model returned empty responses after {max_empty_retries} retries. Finish reason: {last_finish_reason}."
                    yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                    break
                
                # Increment retry counter
                empty_response_retries += 1
                
                # Handle MALFORMED_FUNCTION_CALL specifically - the model tried to call a function but failed
                if is_malformed:
                    malformed_retries += 1
                    if malformed_retries >= max_malformed_retries:
                        logger.error(f"[STREAM] Max malformed function call retries ({max_malformed_retries}) reached. Giving up.")
                        yield json.dumps({"type": "error", "content": f"The model produced {max_malformed_retries} consecutive malformed function calls. Please try rephrasing your question."}) + "\n"
                        break
                    
                    logger.warning(f"[STREAM] Malformed function call detected on turn {current_turn} (retry {malformed_retries}/{max_malformed_retries}). Prompting model to retry properly.")
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
                        # Local counters reset automatically on success
                        break
                    
                    # Still no response - yield error to frontend
                    error_msg = f"The model returned no response. Finish reason: {last_finish_reason}. This may indicate the input is too long or there's a content filtering issue."
                    logger.error(f"[STREAM] {error_msg}")
                    yield json.dumps({"type": "error", "content": error_msg}) + "\n"
                    break
                
                continue
        
        # Stream completed (local retry counters automatically go out of scope)

    except Exception as e:
        logger.exception("Chat stream failed")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
