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

from ..models import ChatRequest, ChatResponse, TokenUsage
from ..tools.filesystem import list_files, read_file
from ..tools.dependencies import list_package_files, read_package_file
from ..search import perform_hybrid_search
from ..active_repo_state import active_repo_state

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

def list_files_wrapper(directory: str = ".") -> str:
    """
    List files in a directory.
    Args:
        directory: Relative path to the directory.
    """
    try:
        base_path = get_active_repo_path()
        # Ensure we don't traverse up
        full_path = os.path.join(base_path, directory)
        return list_files(full_path)
    except Exception as e:
        return f"Error listing files: {e}"

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

# Map tools to functions
tools_map = {
    "rag_search": rag_search,
    "list_files": list_files_wrapper,
    "read_file": read_file_wrapper,
    "list_package_files": list_package_files,
    "read_package_file": read_package_file
}

# Define tool schemas for Vertex AI
rag_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="rag_search",
            description="Search the codebase using RAG. Best for finding code logic, classes, or functions.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="The search query")
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="list_files",
            description="List files in a directory. Use to explore the repo structure.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "directory": types.Schema(type=types.Type.STRING, description="Directory path (default '.')")
                },
                required=["directory"]
            )
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read the content of a file.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "filepath": types.Schema(type=types.Type.STRING, description="Path to the file"),
                    "max_lines": types.Schema(type=types.Type.INTEGER, description="Max lines to read (default 500)")
                },
                required=["filepath"]
            )
        ),
        types.FunctionDeclaration(
            name="list_package_files",
            description="List files in an installed python package (dependency).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "package_name": types.Schema(type=types.Type.STRING, description="Name of the package (e.g. 'fastapi')")
                },
                required=["package_name"]
            )
        ),
        types.FunctionDeclaration(
            name="read_package_file",
            description="Read a specific file from an installed package. Use this after list_package_files to read dependency source code.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "package_name": types.Schema(type=types.Type.STRING, description="Name of the package (e.g. 'chromadb')"),
                    "filepath": types.Schema(type=types.Type.STRING, description="Relative path within package (e.g. 'api/client.py')"),
                    "max_lines": types.Schema(type=types.Type.INTEGER, description="Max lines to read (default 500)")
                },
                required=["package_name", "filepath"]
            )
        )
    ]
)

from ..context_manager import context_manager

def get_system_instruction(context_str: str) -> str:
    """Generate the system instruction with context and CoT protocol."""
    return f"""You are an expert coding assistant for the 'Untango' repository. 

=== AUTOMATED CONTEXT ===
{context_str}
=========================

IMPORTANT: You have access to powerful tools and you MUST use them proactively:
- rag_search: Search the codebase semantically for code logic, functions, and classes
- list_files: Explore the repository structure
- read_file: Read file contents directly
- list_package_files: Inspect installed dependencies
- scan_environment: Get OS, Python version, GPU status, and installed packages
- map_repo: Get a structural map of the repository (entry points, dependencies)
- analyze_notebook: Extract imports and versions from a .ipynb file

CHAIN OF THOUGHT PROTOCOL (REQUIRED):
When you receive a user request, you must follow this reasoning process internally before answering:

1. **Analyze Request**: What is the user asking? What specific information is needed?
2. **Check Context**: Look at the "Automated Context" above. 
   - Do I already know the OS, Python version, or Dependencies?
   - If yes, use that information directly.
3. **Check Sufficiency**: Do I have enough information to answer *completely*?
   - If the user asks about code logic, do I have the code? -> No, use `rag_search`.
   - If the user asks about a specific file, do I have its content? -> No, use `read_file`.
   - If the user asks about "versions", do I need to check for mismatches? -> Yes, check Context + Last Updated Date.
4. **Formulate Plan**:
   - If information is missing, CALL A TOOL.
   - Do NOT ask the user for information you can find yourself.
   - Do NOT answer with "I don't know" without trying to search first.

VERSION INFERENCE (CRITICAL):
If the "Automated Context" shows dependencies without versions (e.g. just "pandas"), you MUST:
1. Check the "Last Updated" date in the [Repository] section.
2. Infer the likely version based on that date (e.g., Repo from 2021 likely uses pandas 1.x).
3. Compare this with the user's installed version.
4. WARN the user if there is a significant time gap (e.g., "Repo is from 2021 but you have pandas 2.2 installed. This may cause breaking changes.").

STRATEGIC SEARCH HEURISTICS:
- **Server configuration**: Look at `main.py` or `config.py`.
- **API endpoints**: Search for `@app.post` / `@app.get`.
- **Data models**: Check `models.py`.
- **Business logic**: Use `rag_search`.

Use `read_file` to inspect actual source code rather than relying only on `rag_search`.
Prefer reading entry point files (`main.py`, `__init__.py`) completely to understand structure.

Only ask the user for clarification if the question is genuinely ambiguous after you have tried to investigate.
"""


async def chat_with_agent(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request using Vertex AI with tools.
    """
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
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

        # Configuration
        config = types.GenerateContentConfig(
            tools=[rag_tool],
            temperature=0.0, # Low temp for tool use
            system_instruction=get_system_instruction(context_str)
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
        current_turn = 0
        
        final_response_text = ""
        usage = None

        while current_turn < max_turns:
            # Check if model wants to call a function
            if not response.candidates:
                break
                
            candidate = response.candidates[0]
            
            # Capture usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                meta = response.usage_metadata
                usage = TokenUsage(
                    input_tokens=getattr(meta, 'prompt_token_count', 0),
                    output_tokens=getattr(meta, 'candidates_token_count', 0),
                    total_tokens=getattr(meta, 'total_token_count', 0)
                )
                input_cost, output_cost, total_cost = calculate_cost(usage.input_tokens, usage.output_tokens)
                logger.info(
                    "Token usage for turn %d: input=%d, output=%d, total=%d, cost=$%.6f (input=$%.6f, output=$%.6f)",
                    current_turn, usage.input_tokens, usage.output_tokens, usage.total_tokens,
                    total_cost, input_cost, output_cost
                )

            # Check for function calls
            function_calls = []
            for part in candidate.content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
            
            if not function_calls:
                # No function calls, this is the final answer
                final_response_text = candidate.content.parts[0].text
                break
            
            # Execute tools
            tool_outputs = []
            contents.append(candidate.content) # Add model's tool call to history
            
            for call in function_calls:
                fn_name = call.name
                fn_args = call.args
                
                logger.info(f"Agent calling tool: {fn_name} with args: {fn_args}")
                
                if fn_name in tools_map:
                    try:
                        # Convert args to dict
                        args_dict = {k: v for k, v in fn_args.items()}
                        # Check if tool is async (not currently, but good practice)
                        result = tools_map[fn_name](**args_dict)
                    except Exception as e:
                        result = f"Error executing {fn_name}: {e}"
                else:
                    result = f"Error: Unknown tool {fn_name}"
                
                tool_outputs.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": result}
                    )
                )
            
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
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
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

        config = types.GenerateContentConfig(
            tools=[rag_tool],
            temperature=0.0,
            system_instruction=get_system_instruction(context_str)
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
            
            # Iterate asynchronously
            async for chunk in response_stream:
                # Check for usage metadata
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    meta = chunk.usage_metadata
                    usage = {
                        "input_tokens": getattr(meta, 'prompt_token_count', 0),
                        "output_tokens": getattr(meta, 'candidates_token_count', 0),
                        "total_tokens": getattr(meta, 'total_token_count', 0)
                    }
                    yield json.dumps({"type": "usage", "usage": usage}) + "\n"

                if not chunk.candidates:
                    continue
                
                candidate = chunk.candidates[0]
                
                for part in candidate.content.parts:
                    if part.text:
                        full_text += part.text
                        yield json.dumps({"type": "token", "content": part.text}) + "\n"
                    
                    if part.function_call:
                        function_calls.append(part.function_call)

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
                    
                    if fn_name in tools_map:
                        try:
                            args_dict = {k: v for k, v in fn_args.items()}
                            result = tools_map[fn_name](**args_dict)
                        except Exception as e:
                            result = f"Error executing {fn_name}: {e}"
                    else:
                        result = f"Error: Unknown tool {fn_name}"
                    
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
                # No function calls, we are done
                if full_text:
                    contents.append(types.Content(role="model", parts=[types.Part(text=full_text)]))
                break

    except Exception as e:
        logger.exception("Chat stream failed")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
