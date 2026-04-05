"""
Metrics Tracking Module
Tracks context drift, error propagation, tool hallucination, and task completion.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StepMetrics:
    """Metrics for a single step in the evaluation loop."""
    step_id: int
    context_drift_score: float
    error_propagation_rate: float
    tool_hallucination_rate: float
    task_completion_status: bool
    tools_used: List[str]
    errors: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all steps."""
    total_steps: int
    avg_context_drift: float
    avg_error_propagation: float
    avg_tool_hallucination: float
    completion_rates: Dict[int, float]  # step -> completion rate
    stability_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsTracker:
    """Tracks and computes all evaluation metrics."""
    
    def __init__(self):
        self.step_metrics: List[StepMetrics] = []
        self.reference_context: str = ""
        self.current_context: str = ""
        self.expected_tools: List[str] = []
        self.error_history: List[str] = []
        self.task_history: List[Dict] = []
    
    def set_reference_context(self, context: str):
        """Set the initial reference context for drift calculation."""
        self.reference_context = context
    
    def set_current_context(self, context: str):
        """Update current context for drift calculation."""
        self.current_context = context
    
    def set_expected_tools(self, tools: List[str]):
        """Set the list of valid/expected tools."""
        self.expected_tools = tools
    
    def compute_context_drift(self) -> float:
        """Compute context drift score using token overlap."""
        if not self.reference_context or not self.current_context:
            return 0.0
        
        ref_tokens = set(self.reference_context.lower().split())
        curr_tokens = set(self.current_context.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        overlap = len(ref_tokens.intersection(curr_tokens))
        drift = 1.0 - (overlap / len(ref_tokens))
        return min(drift, 1.0)
    
    def compute_error_propagation(self, current_errors: List[str]) -> float:
        """Compute error propagation rate."""
        if not current_errors:
            return 0.0
        
        new_errors = [e for e in current_errors if e not in self.error_history]
        propagation_rate = len(new_errors) / max(len(self.error_history), 1) if self.error_history else 0.0
        
        self.error_history.extend(current_errors)
        return min(propagation_rate, 1.0)
    
    def compute_tool_hallucination(self, used_tools: List[str]) -> float:
        """Compute tool hallucination rate (using non-existent tools)."""
        if not used_tools:
            return 0.0
        
        hallucinated = [t for t in used_tools if t not in self.expected_tools]
        rate = len(hallucinated) / len(used_tools)
        return rate
    
    def record_step(self, step_id: int, tools_used: List[str], errors: List[str], 
                    task_completed: bool) -> StepMetrics:
        """Record metrics for a step."""
        context_drift = self.compute_context_drift()
        error_prop = self.compute_error_propagation(errors)
        tool_halluc = self.compute_tool_hallucination(tools_used)
        
        metrics = StepMetrics(
            step_id=step_id,
            context_drift_score=context_drift,
            error_propagation_rate=error_prop,
            tool_hallucination_rate=tool_halluc,
            task_completion_status=task_completed,
            tools_used=tools_used,
            errors=errors
        )
        
        self.step_metrics.append(metrics)
        return metrics
    
    def compute_completion_rate_at_step(self, step: int) -> float:
        """Compute task completion rate at a specific step checkpoint."""
        if not self.step_metrics:
            return 0.0
        
        relevant = [m for m in self.step_metrics if m.step_id <= step]
        if not relevant:
            return 0.0
        
        completed = sum(1 for m in relevant if m.task_completion_status)
        return completed / len(relevant)
    
    def compute_aggregate_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across all steps."""
        if not self.step_metrics:
            return AggregateMetrics(
                total_steps=0,
                avg_context_drift=0.0,
                avg_error_propagation=0.0,
                avg_tool_hallucination=0.0,
                completion_rates={},
                stability_score=0.0
            )
        
        avg_drift = sum(m.context_drift_score for m in self.step_metrics) / len(self.step_metrics)
        avg_error = sum(m.error_propagation_rate for m in self.step_metrics) / len(self.step_metrics)
        avg_halluc = sum(m.tool_hallucination_rate for m in self.step_metrics) / len(self.step_metrics)
        
        completion_rates = {
            10: self.compute_completion_rate_at_step(10),
            20: self.compute_completion_rate_at_step(20),
            30: self.compute_completion_rate_at_step(30),
            40: self.compute_completion_rate_at_step(40),
            50: self.compute_completion_rate_at_step(50)
        }
        
        # Stability score: inverse of average negative metrics
        stability = 1.0 - (avg_drift + avg_error + avg_halluc) / 3.0
        
        return AggregateMetrics(
            total_steps=len(self.step_metrics),
            avg_context_drift=avg_drift,
            avg_error_propagation=avg_error,
            avg_tool_hallucination=avg_halluc,
            completion_rates=completion_rates,
            stability_score=max(0.0, stability)
        )
    
    def get_per_step_logs(self) -> List[Dict[str, Any]]:
        """Get per-step metric logs as JSON-serializable dicts."""
        return [
            {
                "step_id": m.step_id,
                "context_drift_score": m.context_drift_score,
                "error_propagation_rate": m.error_propagation_rate,
                "tool_hallucination_rate": m.tool_hallucination_rate,
                "task_completion_status": m.task_completion_status,
                "tools_used": m.tools_used,
                "errors": m.errors,
                "timestamp": m.timestamp
            }
            for m in self.step_metrics
        ]
    
    def reset(self):
        """Reset all metrics."""
        self.step_metrics = []
        self.reference_context = ""
        self.current_context = ""
        self.expected_tools = []
        self.error_history = []
        self.task_history = []