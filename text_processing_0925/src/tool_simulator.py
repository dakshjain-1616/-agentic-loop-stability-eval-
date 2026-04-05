"""
Tool Simulator Module
Simulates realistic tool interactions without external dependencies.
Supports: file management, code execution, web search, multi-tool chaining.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCall:
    """Represents a tool call made by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    step_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    success: bool
    output: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VirtualFileSystem:
    """Simulates a file system for file management tasks."""
    
    def __init__(self, base_path: str = "/tmp/agent_workspace"):
        self.base_path = base_path
        self.files: Dict[str, str] = {}
        self.structure: Dict[str, Dict] = {}
        self._initialize_default_structure()
    
    def _initialize_default_structure(self):
        """Create initial file system structure."""
        default_files = {
            "config.json": '{"model": "test", "version": 1}',
            "data/sample.txt": "Sample data for processing\nLine 2\nLine 3",
            "scripts/process.py": "def process():\n    return 'processed'",
            "logs/run.log": "Initial log entry",
        }
        for path, content in default_files.items():
            self.write_file(path, content)
    
    def write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            self.files[path] = content
            parts = path.split('/')
            current = self.structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = "file"
            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {path}",
                metadata={"path": path, "size": len(content)}
            )
        except Exception as e:
            return ToolResult(success=False, output="", error_message=str(e))
    
    def read_file(self, path: str) -> ToolResult:
        """Read content from a file."""
        if path not in self.files:
            return ToolResult(success=False, output="", error_message=f"File not found: {path}")
        content = self.files[path]
        return ToolResult(success=True, output=content, metadata={"path": path, "size": len(content)})
    
    def delete_file(self, path: str) -> ToolResult:
        """Delete a file."""
        if path not in self.files:
            return ToolResult(success=False, output="", error_message=f"File not found: {path}")
        del self.files[path]
        return ToolResult(success=True, output=f"Deleted {path}")
    
    def list_directory(self, path: str = "") -> ToolResult:
        """List contents of a directory."""
        files_in_dir = []
        for file_path in self.files.keys():
            if path:
                if file_path.startswith(path + "/"):
                    rel = file_path[len(path)+1:].split('/')[0]
                    if rel not in files_in_dir:
                        files_in_dir.append(rel)
            else:
                first = file_path.split('/')[0]
                if first not in files_in_dir:
                    files_in_dir.append(first)
        return ToolResult(success=True, output="\n".join(sorted(files_in_dir)), metadata={"count": len(files_in_dir)})
    
    def get_state_hash(self) -> str:
        """Get a hash of the current file system state."""
        state_str = json.dumps(sorted(self.files.items()))
        return hashlib.md5(state_str.encode()).hexdigest()


class CodeExecutor:
    """Simulates code execution with safety constraints."""
    
    def __init__(self):
        self.execution_history: List[Dict] = []
    
    def execute(self, code: str, language: str = "python") -> ToolResult:
        """Execute code in a simulated environment."""
        self.execution_history.append({"code": code, "language": language, "timestamp": datetime.now().isoformat()})
        
        if "error" in code.lower() or "raise" in code.lower():
            return ToolResult(success=False, output="", error_message="Simulated execution error")
        
        if "print(" in code:
            import re
            matches = re.findall(r'print\(["\'](.+?)["\']\)', code)
            if matches:
                return ToolResult(success=True, output="\n".join(matches), metadata={"executed": True})
        
        if "return" in code:
            return ToolResult(success=True, output="Function returned value", metadata={"executed": True, "has_return": True})
        
        return ToolResult(success=True, output="Code executed successfully", metadata={"executed": True, "lines": len(code.split('\n'))})


class WebSearchSimulator:
    """Simulates web search with predefined knowledge base."""
    
    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
            "transformers": "Transformers are a type of neural network architecture using self-attention mechanisms.",
            "cuda": "CUDA is a parallel computing platform and API created by NVIDIA for GPU computing.",
            "api": "An API (Application Programming Interface) allows software components to communicate.",
        }
        self.search_history: List[Dict] = []
    
    def search(self, query: str, num_results: int = 3) -> ToolResult:
        """Simulate a web search."""
        self.search_history.append({"query": query, "num_results": num_results, "timestamp": datetime.now().isoformat()})
        query_lower = query.lower()
        results = []
        
        for topic, content in self.knowledge_base.items():
            if topic in query_lower:
                results.append({"title": f"About {topic.title()}", "content": content, "url": f"https://example.com/{topic.replace(' ', '-')}"})
        
        if len(results) < num_results:
            for topic in list(self.knowledge_base.keys())[len(results):num_results]:
                results.append({"title": f"About {topic.title()}", "content": self.knowledge_base[topic], "url": f"https://example.com/{topic}"})
        
        return ToolResult(success=True, output=json.dumps(results[:num_results], indent=2), metadata={"results_count": len(results[:num_results])})


class ToolSimulator:
    """Main tool simulator that orchestrates all tool types."""
    
    def __init__(self):
        self.file_system = VirtualFileSystem()
        self.code_executor = CodeExecutor()
        self.web_search = WebSearchSimulator()
        self.call_history: List[ToolCall] = []
        self.result_history: List[ToolResult] = []
        self.initial_state_hash = self.file_system.get_state_hash()
    
    def get_available_tools(self) -> List[str]:
        return ["read_file", "write_file", "delete_file", "list_directory", "execute_code", "web_search", "get_file_state"]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], step_id: int) -> ToolResult:
        """Execute a tool call and record the result."""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments, step_id=step_id)
        self.call_history.append(tool_call)
        
        try:
            if tool_name == "read_file":
                result = self.file_system.read_file(arguments.get("path", ""))
            elif tool_name == "write_file":
                result = self.file_system.write_file(arguments.get("path", ""), arguments.get("content", ""))
            elif tool_name == "delete_file":
                result = self.file_system.delete_file(arguments.get("path", ""))
            elif tool_name == "list_directory":
                result = self.file_system.list_directory(arguments.get("path", ""))
            elif tool_name == "execute_code":
                result = self.code_executor.execute(arguments.get("code", ""), arguments.get("language", "python"))
            elif tool_name == "web_search":
                result = self.web_search.search(arguments.get("query", ""), arguments.get("num_results", 3))
            elif tool_name == "get_file_state":
                current_hash = self.file_system.get_state_hash()
                result = ToolResult(success=True, output=f"State hash: {current_hash}, Changed: {current_hash != self.initial_state_hash}", metadata={"hash": current_hash, "changed": current_hash != self.initial_state_hash})
            else:
                result = ToolResult(success=False, output="", error_message=f"Unknown tool: {tool_name}")
        except Exception as e:
            result = ToolResult(success=False, output="", error_message=str(e))
        
        self.result_history.append(result)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        tool_counts = {}
        for call in self.call_history:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
        success_count = sum(1 for r in self.result_history if r.success)
        return {
            "total_calls": len(self.call_history),
            "success_rate": success_count / max(len(self.result_history), 1),
            "tool_distribution": tool_counts,
            "file_system_hash": self.file_system.get_state_hash(),
            "state_changed": self.file_system.get_state_hash() != self.initial_state_hash
        }
    
    def reset(self):
        self.file_system = VirtualFileSystem()
        self.code_executor = CodeExecutor()
        self.web_search = WebSearchSimulator()
        self.call_history = []
        self.result_history = []
        self.initial_state_hash = self.file_system.get_state_hash()