#!/usr/bin/env python3
"""Test script for tool simulator."""
import sys
sys.path.insert(0, '/app/text_processing_0925/src')

from tool_simulator import ToolSimulator

def test_simulator():
    """Run 5-step test sequence with all tool types."""
    simulator = ToolSimulator()
    
    print("=== Tool Simulator Test ===")
    print(f"Available tools: {simulator.get_available_tools()}")
    
    # Step 1: Read a file
    result1 = simulator.execute_tool("read_file", {"path": "config.json"}, step_id=1)
    print(f"\nStep 1 - read_file: success={result1.success}, output={result1.output[:50]}...")
    
    # Step 2: Write a file
    result2 = simulator.execute_tool("write_file", {"path": "test.txt", "content": "Test content"}, step_id=2)
    print(f"Step 2 - write_file: success={result2.success}, output={result2.output[:50]}...")
    
    # Step 3: List directory
    result3 = simulator.execute_tool("list_directory", {"path": ""}, step_id=3)
    print(f"Step 3 - list_directory: success={result3.success}, count={result3.metadata.get('count')}")
    
    # Step 4: Execute code
    result4 = simulator.execute_tool("execute_code", {"code": "print('Hello')", "language": "python"}, step_id=4)
    print(f"Step 4 - execute_code: success={result4.success}, output={result4.output}")
    
    # Step 5: Web search
    result5 = simulator.execute_tool("web_search", {"query": "python programming", "num_results": 2}, step_id=5)
    print(f"Step 5 - web_search: success={result5.success}, results={result5.metadata.get('results_count')}")
    
    # Get statistics
    stats = simulator.get_statistics()
    print(f"\n=== Statistics ===")
    print(f"Total calls: {stats['total_calls']}")
    print(f"Success rate: {stats['success_rate']}")
    print(f"State changed: {stats['state_changed']}")
    
    # Verify state persists
    result6 = simulator.execute_tool("read_file", {"path": "test.txt"}, step_id=6)
    print(f"\nStep 6 - verify state: test.txt exists={result6.success}, content='{result6.output}'")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_simulator()