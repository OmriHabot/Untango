"""
AST Tools
Provides enhanced code structure analysis using Python's AST module.
Enables finding function usages, class hierarchies, and call graphs.
"""
import os
import ast
import logging
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Directories to skip when scanning
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'node_modules', '.repos'}


def get_python_files(repo_path: str) -> List[str]:
    """Get all Python files in the repository."""
    python_files = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for f in files:
            if f.endswith('.py'):
                python_files.append(os.path.join(root, f))
    
    return python_files


def parse_file(filepath: str) -> Optional[ast.AST]:
    """Parse a Python file and return its AST."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return ast.parse(f.read(), filename=filepath)
    except SyntaxError as e:
        logger.debug(f"Syntax error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Failed to parse {filepath}: {e}")
        return None


def find_function_usages(
    repo_path: str,
    function_name: str,
    include_definitions: bool = False
) -> Dict[str, Any]:
    """
    Find all places where a function is called in the codebase.
    
    Args:
        repo_path: Path to the repository
        function_name: Name of the function to search for
        include_definitions: If True, also include function definitions
        
    Returns:
        Dict with usages list and count
    """
    result = {
        "success": False,
        "function_name": function_name,
        "usages": [],
        "definitions": [],
        "usage_count": 0,
    }
    
    class FunctionUsageFinder(ast.NodeVisitor):
        def __init__(self, filepath: str):
            self.filepath = filepath
            self.usages = []
            self.definitions = []
        
        def visit_Call(self, node: ast.Call):
            # Direct function call: function_name()
            if isinstance(node.func, ast.Name) and node.func.id == function_name:
                self.usages.append({
                    "file": self.filepath,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "type": "direct_call",
                })
            
            # Method call: obj.function_name()
            elif isinstance(node.func, ast.Attribute) and node.func.attr == function_name:
                self.usages.append({
                    "file": self.filepath,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "type": "method_call",
                })
            
            self.generic_visit(node)
        
        def visit_FunctionDef(self, node: ast.FunctionDef):
            if node.name == function_name:
                self.definitions.append({
                    "file": self.filepath,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "decorators": [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list],
                })
            self.generic_visit(node)
        
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            if node.name == function_name:
                self.definitions.append({
                    "file": self.filepath,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "async": True,
                })
            self.generic_visit(node)
    
    python_files = get_python_files(repo_path)
    
    for filepath in python_files:
        tree = parse_file(filepath)
        if tree is None:
            continue
        
        rel_path = os.path.relpath(filepath, repo_path)
        finder = FunctionUsageFinder(rel_path)
        finder.visit(tree)
        
        result["usages"].extend(finder.usages)
        result["definitions"].extend(finder.definitions)
    
    result["usage_count"] = len(result["usages"])
    result["success"] = True
    
    if not include_definitions:
        result.pop("definitions", None)
    
    return result


def get_function_details(
    repo_path: str,
    filepath: str,
    function_name: str
) -> Dict[str, Any]:
    """
    Get detailed information about a function.
    
    Args:
        repo_path: Path to the repository
        filepath: Path to the file containing the function
        function_name: Name of the function
        
    Returns:
        Dict with function signature, docstring, decorators, etc.
    """
    result = {
        "success": False,
        "name": function_name,
        "filepath": filepath,
        "signature": "",
        "docstring": "",
        "decorators": [],
        "line_start": 0,
        "line_end": 0,
        "is_async": False,
        "parameters": [],
        "returns": None,
    }
    
    full_path = os.path.join(repo_path, filepath)
    tree = parse_file(full_path)
    
    if tree is None:
        result["error"] = f"Could not parse file: {filepath}"
        return result
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                result["success"] = True
                result["line_start"] = node.lineno
                result["line_end"] = node.end_lineno
                result["is_async"] = isinstance(node, ast.AsyncFunctionDef)
                
                # Get docstring
                docstring = ast.get_docstring(node)
                if docstring:
                    result["docstring"] = docstring
                
                # Get decorators
                for dec in node.decorator_list:
                    if hasattr(ast, 'unparse'):
                        result["decorators"].append(ast.unparse(dec))
                    else:
                        result["decorators"].append(str(dec))
                
                # Get parameters
                for arg in node.args.args:
                    param = {"name": arg.arg}
                    if arg.annotation and hasattr(ast, 'unparse'):
                        param["type"] = ast.unparse(arg.annotation)
                    result["parameters"].append(param)
                
                # Get return type
                if node.returns and hasattr(ast, 'unparse'):
                    result["returns"] = ast.unparse(node.returns)
                
                # Build signature
                params = []
                for p in result["parameters"]:
                    if "type" in p:
                        params.append(f"{p['name']}: {p['type']}")
                    else:
                        params.append(p['name'])
                
                ret = f" -> {result['returns']}" if result["returns"] else ""
                async_prefix = "async " if result["is_async"] else ""
                result["signature"] = f"{async_prefix}def {function_name}({', '.join(params)}){ret}"
                
                break
    
    return result


def get_class_hierarchy(
    repo_path: str,
    class_name: str
) -> Dict[str, Any]:
    """
    Get the inheritance hierarchy for a class.
    
    Args:
        repo_path: Path to the repository
        class_name: Name of the class to analyze
        
    Returns:
        Dict with base classes, subclasses, and methods
    """
    result = {
        "success": False,
        "class_name": class_name,
        "definition": None,
        "base_classes": [],
        "subclasses": [],
        "methods": [],
    }
    
    all_classes = {}  # class_name -> (filepath, base_classes)
    
    python_files = get_python_files(repo_path)
    
    for filepath in python_files:
        tree = parse_file(filepath)
        if tree is None:
            continue
        
        rel_path = os.path.relpath(filepath, repo_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                
                all_classes[node.name] = {
                    "file": rel_path,
                    "line": node.lineno,
                    "bases": bases,
                }
                
                # If this is our target class, get details
                if node.name == class_name:
                    result["definition"] = {
                        "file": rel_path,
                        "line": node.lineno,
                        "end_line": node.end_lineno,
                    }
                    result["base_classes"] = bases
                    
                    # Get docstring
                    docstring = ast.get_docstring(node)
                    if docstring:
                        result["definition"]["docstring"] = docstring
                    
                    # Get methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {
                                "name": item.name,
                                "line": item.lineno,
                                "is_async": isinstance(item, ast.AsyncFunctionDef),
                            }
                            
                            # Check for property/classmethod/staticmethod
                            for dec in item.decorator_list:
                                if isinstance(dec, ast.Name):
                                    if dec.id == "property":
                                        method_info["is_property"] = True
                                    elif dec.id == "classmethod":
                                        method_info["is_classmethod"] = True
                                    elif dec.id == "staticmethod":
                                        method_info["is_staticmethod"] = True
                            
                            result["methods"].append(method_info)
                    
                    result["success"] = True
    
    # Find subclasses
    for name, info in all_classes.items():
        if class_name in info["bases"]:
            result["subclasses"].append({
                "name": name,
                "file": info["file"],
                "line": info["line"],
            })
    
    return result


def get_imports_in_file(filepath: str) -> Dict[str, Any]:
    """
    Get all imports in a Python file.
    
    Returns:
        Dict with imports list
    """
    result = {
        "success": False,
        "imports": [],
        "from_imports": [],
    }
    
    tree = parse_file(filepath)
    if tree is None:
        return result
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append({
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                result["from_imports"].append({
                    "module": node.module or "",
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "level": node.level,  # relative import level
                })
    
    result["success"] = True
    return result


def get_file_symbols(
    repo_path: str,
    filepath: str
) -> Dict[str, Any]:
    """
    Get all symbols (functions, classes, variables) defined in a file.
    
    Returns:
        Dict with functions, classes, and global variables
    """
    result = {
        "success": False,
        "functions": [],
        "classes": [],
        "global_variables": [],
    }
    
    full_path = os.path.join(repo_path, filepath)
    tree = parse_file(full_path)
    
    if tree is None:
        return result
    
    # Only look at top-level definitions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "is_async": False,
            })
        elif isinstance(node, ast.AsyncFunctionDef):
            result["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "is_async": True,
            })
        elif isinstance(node, ast.ClassDef):
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
            })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result["global_variables"].append({
                        "name": target.id,
                        "line": node.lineno,
                    })
    
    result["success"] = True
    return result
