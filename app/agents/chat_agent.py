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

# Map tools to functions
# Map tools to functions
tools_map = {
    "rag_search": rag_search,
    "read_file": read_file_wrapper,
    "list_files": list_files_wrapper,
    "get_active_repo_path": get_active_repo_path,
    # get_system_instruction will be added after definition
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
            name="read_file",
            description="Read the full content of a file. Use this to examine source code, configs, or any text file.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "filepath": types.Schema(type=types.Type.STRING, description="Relative path to the file from repo root"),
                    "max_lines": types.Schema(type=types.Type.INTEGER, description="Max lines to read (default 500)")
                },
                required=["filepath"]
            )
        ),
        types.FunctionDeclaration(
            name="list_files",
            description="List all files and subdirectories in a directory. Use this to explore the repository structure and find files to investigate.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "directory": types.Schema(type=types.Type.STRING, description="Relative path to directory (default: repo root)")
                },
                required=[]
            )
        ),
        types.FunctionDeclaration(
            name="get_system_instruction",
            description="Get the system instruction used by the agent. Useful for self-reflection or understanding current instructions.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "context_details": types.Schema(type=types.Type.STRING, description="Optional context details to include in the instruction.")
                },
                required=[]
            )
        ),
        types.FunctionDeclaration(
            name="get_active_repo_path",
            description="Get the absolute path of the currently active repository.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
                required=[]
            )
        ),
        types.FunctionDeclaration(
            name="get_context_report",
            description="Get the automated context report (Environment, Repository, Dependencies).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
                required=[]
            )
        )
    ]
)

from ..context_manager import context_manager

def get_system_instruction(context_str: str, repo_name: str) -> str:
    """Generate the system instruction with context and CoT protocol."""
    return f"""You are an expert software developer and coding assistant for the '{repo_name}' repository.
You have full access to the source code of '{repo_name}' and are answering questions specifically about it.

=== AUTOMATED CONTEXT ===
{context_str}
=========================

=== MANDATORY FIRST STEP ===

**ALWAYS call `list_files()` FIRST before doing anything else!**
This gives you a complete view of all files in the repository.
Then drill into subdirectories with `list_files("subdir")` as needed.

=== YOUR TOOLS (USE THEM!) ===

You have access to these tools to explore and understand the codebase. Use them liberally!

**1. list_files(directory=".")**
   - Lists all files and subdirectories in a directory
   - ALWAYS call this first to see the full repo structure
   - Then drill into subdirectories: `list_files("src")`, `list_files("tests")`, etc.

**2. read_file(filepath, max_lines=500)**  
   - Reads the full content of any file
   - Use this to examine source code, configs, README, etc.
   - Path is relative to repo root (e.g., "src/main.py")
   - USE THIS LIBERALLY - read every file you're curious about

**3. rag_search(query)**
   - Semantic search across the entire codebase
   - Best for finding: function implementations, class definitions, usage patterns
   - Example queries: "how is authentication handled", "database connection", "error handling"

**4. get_context_report()**
   - Returns environment info (OS, Python version), repo structure, and dependencies
   - Already provided above, but call this if you need a refresh

**5. get_active_repo_path()**
   - Returns the absolute filesystem path to the repository
   - Useful if you need to understand file paths

=== EXPLORATION WORKFLOW ===

When answering any question, follow this EXACT workflow:

1. **FIRST: Call `list_files()`** (MANDATORY!)
   - This shows you ALL files in the repository root
   - Then call `list_files("subdir")` for any interesting subdirectories

2. **Read entry points**:
   - main.py, app.py, index.js, __init__.py
   - README.md for project overview
   - config.py, settings.py for configuration

3. **Follow the trail**:
   - When you see imports, read those files too
   - When you find a function call, search for its definition
   - When you see a class, read its full implementation

4. **Search for patterns**:
   - Use `rag_search` to find all usages of a function/class
   - Search for error messages, config keys, API endpoints

5. **Read related files**:
   - Tests often reveal expected behavior
   - Documentation files explain intent
   - Config files reveal available options

=== KEEP EXPLORING UNTIL SATISFIED ===

If you check a directory or file and don't find what you need:
- **Try another directory** - keep calling `list_files()` on different folders
- **Try a different search query** - rephrase and use `rag_search` again
- **Read more files** - the answer might be in a related file you haven't checked
- **Don't give up early** - explore until you have enough information

You have unlimited tool calls. Use them! Keep exploring until you're confident in your answer.

=== CRITICAL RULES ===

**DO:**
- Use `read_file` on EVERY file you're curious about
- Use `list_files` to explore unfamiliar directories
- Use `rag_search` to find function/class usages
- Read MORE files rather than fewer
- Follow import chains completely

**DON'T:**
- NEVER answer "I don't have access to..." - you DO have access, use the tools!
- NEVER say "I would need to see..." - just go read it!
- NEVER guess what a file contains - read it!
- NEVER assume based on filename alone - read it!
- NEVER give incomplete answers when you could read one more file

=== SELF-CHECK BEFORE ANSWERING ===

Before writing your final answer, ask yourself:
1. "Have I seen the actual code, or am I guessing?"
2. "Is there another file I should check?"
3. "Did I follow all the imports?"
4. "Would reading one more file improve my answer?"

If YES to any of these → investigate more first!

=== VERSION INFERENCE ===
If dependencies lack versions, check the "Last Updated" date and infer versions accordingly.
WARN the user if there's a significant time gap between repo age and their current packages.
"""

def get_system_instruction_wrapper(context_details: str = "") -> str:
    """Wrapper for get_system_instruction to be used as a tool."""
    return get_system_instruction(context_details)

def get_context_report_wrapper() -> str:
    """Wrapper for get_context_report to be used as a tool."""
    report = context_manager.get_context_report()
    return report.to_string() if report else "Context not initialized."

# Add to tools map
tools_map["get_system_instruction"] = get_system_instruction_wrapper
tools_map["get_context_report"] = get_context_report_wrapper


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

        # Configuration
        config = types.GenerateContentConfig(
            tools=[rag_tool],
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

        config = types.GenerateContentConfig(
            tools=[rag_tool],
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
