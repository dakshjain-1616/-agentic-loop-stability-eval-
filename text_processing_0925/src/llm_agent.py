"""
LLM Agent Module - Wraps local LLM inference using transformers + CUDA
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AgentResponse:
    text: str
    tool_calls: List[Dict[str, Any]]
    confidence: float
    step_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMAgent:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", use_cuda: bool = True):
        self.model_name = model_name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = None
        self.tokenizer = None
        self.context_history: List[str] = []
        self.max_steps = 50
        
        if self.use_cuda:
            print("Using CUDA: " + torch.cuda.get_device_name(0))
        else:
            print("CUDA not available, using CPU")
    
    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("Loading model: " + self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_cuda else torch.float32,
                device_map="auto" if self.use_cuda else None
            )
            
            if not self.use_cuda:
                self.model = self.model.to(self.device)
            
            print("Model loaded on " + str(self.device))
            return True
        except Exception as e:
            print("Error loading model: " + str(e))
            return False
    
    def format_prompt(self, task_description: str, context: str, step_id: int, 
                      available_tools: List[str], previous_actions: List[Dict]) -> str:
        tools_str = ", ".join(available_tools)
        prev_json = json.dumps(previous_actions[:3]) if previous_actions else "None"
        
        prompt = "You are an AI agent that can use tools to complete tasks.\n"
        prompt += "Available tools: " + tools_str + "\n\n"
        prompt += "Task: " + task_description + "\n"
        prompt += "Context: " + context + "\n"
        prompt += "Step: " + str(step_id) + "/" + str(self.max_steps) + "\n\n"
        prompt += "Previous actions: " + prev_json + "\n\n"
        prompt += "Respond in this format:\n"
        prompt += "TOOL: tool_name\n"
        prompt += "ARGS: {arg1: value1}\n"
        prompt += "THOUGHT: Your reasoning\n\n"
        prompt += "If task is complete, respond: COMPLETE: Task finished successfully\n"
        
        return prompt
    
    def parse_response(self, response_text: str, step_id: int) -> AgentResponse:
        tool_calls = []
        confidence = 0.5
        
        lines = response_text.strip().split("\n")
        for line in lines:
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
                tool_calls.append({"tool_name": tool_name, "arguments": {}})
                confidence = 0.8
            elif line.startswith("COMPLETE:"):
                confidence = 1.0
        
        return AgentResponse(text=response_text, tool_calls=tool_calls, confidence=confidence, step_id=step_id)
    
    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            return "Model not loaded"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if self.use_cuda:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):]
        except Exception as e:
            return "Error generating response: " + str(e)
    
    def execute_step(self, task: Any, context: str, step_id: int, 
                     available_tools: List[str]) -> AgentResponse:
        prompt = self.format_prompt(
            task.description, context, step_id, available_tools,
            self.context_history[-5:] if self.context_history else []
        )
        
        response_text = self.generate_response(prompt)
        parsed = self.parse_response(response_text, step_id)
        
        self.context_history.append(response_text)
        return parsed
    
    def get_context_length(self) -> int:
        return sum(len(c) for c in self.context_history)
    
    def reset(self):
        self.context_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "device": str(self.device),
            "cuda_enabled": self.use_cuda,
            "context_length": self.get_context_length(),
            "history_length": len(self.context_history)
        }