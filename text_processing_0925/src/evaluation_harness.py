"""
Agentic Evaluation Harness
Runs LLM through 50-step tool-use loops and measures stability metrics.
"""

import sys
import os
import json
import torch
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool_simulator import ToolSimulator
from metrics import MetricsTracker
from task_generator import TaskGenerator, BenchmarkTask
from llm_agent import LLMAgent


class EvaluationHarness:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", use_cuda: bool = True):
        self.tool_simulator = ToolSimulator()
        self.metrics_tracker = MetricsTracker()
        self.task_generator = TaskGenerator()
        self.agent = LLMAgent(model_name=model_name, use_cuda=use_cuda)
        self.results: Dict[str, Any] = {}
        self.output_dir = "/app/text_processing_0925/outputs"
        self.benchmark_dir = "/app/text_processing_0925/benchmarks"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.benchmark_dir, exist_ok=True)
    
    def run_step(self, task: BenchmarkTask, step_id: int, context: str) -> Dict[str, Any]:
        available_tools = self.tool_simulator.get_available_tools()
        response = self.agent.execute_step(task, context, step_id, available_tools)
        
        tools_used = []
        errors = []
        task_completed = False
        
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("tool_name", "")
            args = tool_call.get("arguments", {})
            result = self.tool_simulator.execute_tool(tool_name, args, step_id)
            tools_used.append(tool_name)
            if not result.success:
                errors.append(result.error_message or "Tool execution failed")
            if tool_name == "COMPLETE":
                task_completed = True
        
        new_context = context + " | Step " + str(step_id) + ": Used " + str(tools_used)
        self.metrics_tracker.set_current_context(new_context)
        self.metrics_tracker.record_step(step_id=step_id, tools_used=tools_used, errors=errors, task_completed=task_completed or (step_id >= task.steps_required))
        
        return {"step_id": step_id, "tools_used": tools_used, "errors": errors, "task_completed": task_completed, "response_confidence": response.confidence}
    
    def run_evaluation(self, num_steps: int = 50) -> Dict[str, Any]:
        print("Starting " + str(num_steps) + "-step evaluation...")
        print("Model: " + self.agent.model_name)
        print("Device: " + str(self.agent.device))
        
        if not self.agent.load_model():
            return {"error": "Failed to load model"}
        
        tasks = self.task_generator.generate_50_step_sequence()
        self.metrics_tracker.set_expected_tools(self.tool_simulator.get_available_tools())
        
        step_results = []
        current_step = 0
        
        for task in tasks:
            self.metrics_tracker.set_reference_context(task.initial_context)
            self.agent.reset()
            
            for step in range(task.steps_required):
                if current_step >= num_steps:
                    break
                context = self.task_generator.generated_tasks[0].initial_context if self.task_generator.generated_tasks else "Default context"
                result = self.run_step(task, current_step + 1, context)
                step_results.append(result)
                current_step += 1
        
        aggregate = self.metrics_tracker.compute_aggregate_metrics()
        
        self.results = {
            "total_steps": current_step,
            "aggregate_metrics": {
                "avg_context_drift": aggregate.avg_context_drift,
                "avg_error_propagation": aggregate.avg_error_propagation,
                "avg_tool_hallucination": aggregate.avg_tool_hallucination,
                "stability_score": aggregate.stability_score,
                "completion_rates": aggregate.completion_rates
            },
            "step_results": step_results,
            "task_statistics": self.task_generator.get_task_statistics(),
            "agent_statistics": self.agent.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }
        
        return self.results
    
    def save_outputs(self):
        step_logs = self.metrics_tracker.get_per_step_logs()
        with open(os.path.join(self.output_dir, "per_step_logs.json"), "w") as f:
            json.dump(step_logs, f, indent=2)
        
        with open(os.path.join(self.output_dir, "stability_report.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        
        benchmark_data = {
            "tasks": [{"task_id": t.task_id, "task_type": t.task_type, "description": t.description, "difficulty": t.difficulty, "steps_required": t.steps_required} for t in self.task_generator.generated_tasks],
            "metrics_summary": self.results.get("aggregate_metrics", {}),
            "generated_at": datetime.now().isoformat()
        }
        with open(os.path.join(self.benchmark_dir, "benchmark_dataset.json"), "w") as f:
            json.dump(benchmark_data, f, indent=2)
        
        print("Outputs saved to " + self.output_dir + " and " + self.benchmark_dir)
    
    def print_summary(self):
        print("")
        print("=== EVALUATION SUMMARY ===")
        print("Total steps executed: " + str(self.results.get('total_steps', 0)))
        print("Stability score: " + str(round(self.results.get('aggregate_metrics', {}).get('stability_score', 0), 3)))
        print("Avg context drift: " + str(round(self.results.get('aggregate_metrics', {}).get('avg_context_drift', 0), 3)))
        print("Avg error propagation: " + str(round(self.results.get('aggregate_metrics', {}).get('avg_error_propagation', 0), 3)))
        print("Avg tool hallucination: " + str(round(self.results.get('aggregate_metrics', {}).get('avg_tool_hallucination', 0), 3)))
        print("")
        print("Completion rates:")
        for step, rate in self.results.get('aggregate_metrics', {}).get('completion_rates', {}).items():
            print("  Step " + str(step) + ": " + str(round(rate, 2)))


def main():
    harness = EvaluationHarness(model_name="Qwen/Qwen3.5-0.8B", use_cuda=torch.cuda.is_available())
    results = harness.run_evaluation(num_steps=50)
    harness.save_outputs()
    harness.print_summary()
    return results


if __name__ == "__main__":
    main()