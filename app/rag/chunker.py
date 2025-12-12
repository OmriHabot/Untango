"""
Code chunking utilities using AST parsing.
"""
import ast
from typing import List, Dict, Any
from fastapi import HTTPException

from ..core.models import ChunkMetadata


def chunk_python_code(code: str, filepath: str, repo_name: str) -> List[Dict[str, Any]]:
    """
    intelligently chunk python code using ast parsing.
    this approach respects code structure and creates semantically coherent chunks.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"syntax error in code: {e}")
    
    chunks = []
    chunk_id = 0
    
    # extract imports at module level
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"from {module} import {alias.name}")
    
    # convert imports list to comma-separated string for chromadb compatibility
    imports_str = ", ".join(imports) if imports else ""
    
    # process top-level nodes
    for node in tree.body:
        chunk_start = node.lineno
        chunk_end = node.end_lineno if hasattr(node, 'end_lineno') else chunk_start
        
        if isinstance(node, ast.FunctionDef):
            # function-level chunking
            chunk_content = ast.get_source_segment(code, node)
            metadata = ChunkMetadata(
                filepath=filepath,
                repo_name=repo_name,
                chunk_type="function",
                start_line=chunk_start,
                end_line=chunk_end,
                function_name=node.name,
                imports=imports_str  # pass string instead of list
            )
            chunks.append({
                "id": f"{filepath}::func::{node.name}::{chunk_id}",
                "content": chunk_content,
                "metadata": metadata.dict()
            })
            chunk_id += 1
            
        elif isinstance(node, ast.ClassDef):
            # class-level chunking with methods
            class_content = ast.get_source_segment(code, node)
            metadata = ChunkMetadata(
                filepath=filepath,
                repo_name=repo_name,
                chunk_type="class",
                start_line=chunk_start,
                end_line=chunk_end,
                class_name=node.name,
                imports=imports_str  # pass string instead of list
            )
            chunks.append({
                "id": f"{filepath}::class::{node.name}::{chunk_id}",
                "content": class_content,
                "metadata": metadata.dict()
            })
            chunk_id += 1
            
            # also chunk individual methods within the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_content = ast.get_source_segment(code, item)
                    method_metadata = ChunkMetadata(
                        filepath=filepath,
                        repo_name=repo_name,
                        chunk_type="method",
                        start_line=item.lineno,
                        end_line=item.end_lineno if hasattr(item, 'end_lineno') else item.lineno,
                        function_name=f"{node.name}.{item.name}",
                        class_name=node.name,
                        imports=imports_str  # pass string instead of list
                    )
                    chunks.append({
                        "id": f"{filepath}::method::{node.name}.{item.name}::{chunk_id}",
                        "content": method_content,
                        "metadata": method_metadata.dict()
                    })
                    chunk_id += 1
        else:
            # top-level statements (assignments, etc.)
            chunk_content = ast.get_source_segment(code, node)
            if chunk_content and chunk_content.strip():
                metadata = ChunkMetadata(
                    filepath=filepath,
                    repo_name=repo_name,
                    chunk_type="top_level",
                    start_line=chunk_start,
                    end_line=chunk_end,
                    imports=imports_str  # pass string instead of list
                )
                chunks.append({
                    "id": f"{filepath}::top::{chunk_id}",
                    "content": chunk_content,
                    "metadata": metadata.dict()
                })
                chunk_id += 1
    
    return chunks

