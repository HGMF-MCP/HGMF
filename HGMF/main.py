#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import os
import sys
import json
import time
import datetime
import re
from typing import List, Dict, Any, Tuple, Optional

try:
    import pynvml
    GPU_MONITOR_AVAILABLE = True
except ImportError:
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")
    GPU_MONITOR_AVAILABLE = False

from sampler import ToolSampler, load_sampler_config
from matcher import ToolMatcher
from utils import generate_grid_search_params, generate_user_query_from_template

MODEL_NAME = "gemma:7b"

def get_gpu_memory_usage():
    if not GPU_MONITOR_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(2)  # GPU 2
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'used_gb': info.used / 1024**3,
            'total_gb': info.total / 1024**3,
            'free_gb': info.free / 1024**3,
            'utilization_percent': (info.used / info.total) * 100
        }
    except Exception as e:
        print(f"GPU Monitoring Error: {e}")
        return None


def print_memory_usage(stage_name: str):
    """
    Print current GPU memory usage.
    Args:
        stage_name: Stage name for logging
    """
    memory_info = get_gpu_memory_usage()
    if memory_info:
        print(f"[{stage_name}] VRAM used: {memory_info['used_gb']:.2f}GB "
              f"({memory_info['utilization_percent']:.1f}%) "
              f"free: {memory_info['free_gb']:.2f}GB")
    else:
        print(f"[{stage_name}] GPU monitoring unavailable")


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_tool_assistant(text: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = re.compile(
        r'<tool_assistant>\s*server:\s*(.*?)\s*tool:\s*(.*?)\s*</tool_assistant>',
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        server_desc = match.group(1).strip()
        tool_desc = match.group(2).strip()
        return server_desc, tool_desc
    return None, None

def chat_ollama(system_prompt, user_prompt, model_name=MODEL_NAME):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    resp = requests.post(url, json=payload)
    result = resp.json()
    return result["message"]["content"]

def test_llm_retrieval(
    sampled_data: List[Dict[str, Any]],
    target_server: Dict[str, Any],
    target_tool: Dict[str, Any],
    sample_size: int = 20,
    position_index: int = 0,
    use_random_selection: bool = False,
    output_dir: Optional[str] = None,
    model_name: str = MODEL_NAME  # default use global
) -> Tuple[Dict[str, Any], Optional[int]]:
    # Log initial VRAM usage
    print_memory_usage("Test start")
    """
    Args:
        sampled_data: Sampled data
        target_server: Target server info
        target_tool: Target tool info
        sample_size: Number of sampled tools
        position_index: Target tool position index
        use_random_selection: Whether to use random selection
        output_dir: Output directory path
        model_name: Model name
    Returns:
        (result, None)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
    # Build result filename suffix
    if use_random_selection:
        selection_method = "random"
    else:
        selection_method = f"position_index_{position_index}"
    
    # Read system prompt
    system_prompt_path = os.path.join(output_dir, "system_ours_mcptools.prompt")
    system_prompt = read_text_file(system_prompt_path)
    
    # Read user prompt template
    user_prompt_template_path = os.path.join(output_dir, "user_query_with_server.prompt")
    user_prompt_template = read_text_file(user_prompt_template_path)
    
    # Get target descriptions
    server_description = target_server.get("description", "")
    tool_description = target_tool.get("description", "")
    
    # Build user prompt
    user_prompt = user_prompt_template.replace("{server_description}", server_description)
    user_prompt = user_prompt.replace("{tool_description}", tool_description)
    
    # Timing
    start_time = time.time()
    
    # VRAM before LLM call
    print_memory_usage("Before LLM call")
    
    # Call LLM
    try:
        response = chat_ollama(system_prompt, user_prompt, model_name=model_name)
        success = True
    except Exception as e:
        print(f"Error: {str(e)}")
        response = f"Error: {str(e)}"
        success = False
    
    # VRAM after LLM call
    print_memory_usage("After LLM call")
    
    # Timing end
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Extract descriptions from response
    extracted_server_desc, extracted_tool_desc = extract_tool_assistant(response)
    
    # Initialize match results
    is_correct = False
    matched_server = None
    matched_tool = None
    
    TOP_SERVERS = 5
    TOP_TOOLS = 1
    
    # If extracted, perform vector matching
    if extracted_server_desc and extracted_tool_desc:
        # VRAM before matching
        print_memory_usage("Vector matching start")
        
        # Init matcher
        matcher = ToolMatcher(top_servers=TOP_SERVERS, top_tools=TOP_TOOLS)
        
        # Setup OpenAI client (same config as data/mcp-tools)
        base_url = ""
        api_version = ""
        api_key = ""
        matcher.setup_openai_client(base_url, api_version, api_key)
        
        # Prepare match result
        match_result = {
            "success": True,
            "server_description": extracted_server_desc,
            "task_description": extracted_tool_desc,
            "matched_servers": [],
            "matched_tools": []
        }
        
        try:
            query_server_embedding = matcher.get_embedding(extracted_server_desc)
            query_tool_embedding = matcher.get_embedding(extracted_tool_desc)
            
            if not query_server_embedding or not query_tool_embedding:
                raise ValueError("Failed to get embeddings for server or tool description")
            
            # Stage 1: match servers
            server_scores = []
            for server in sampled_data:
                if "server_description_embedding" not in server:
                    continue
                if "tools" not in server or not server["tools"]:
                    continue
                
                desc_similarity = matcher.cosine_similarity(
                    query_server_embedding, 
                    server["server_description_embedding"]
                )
                
                summary_similarity = 0
                if "server_summary_embedding" in server:
                    summary_similarity = matcher.cosine_similarity(
                        query_server_embedding, 
                        server["server_summary_embedding"]
                    )
                
                final_score = max(desc_similarity, summary_similarity)
                
                server_scores.append({
                    "server": server,
                    "score": final_score
                })
            
            server_scores.sort(key=lambda x: x["score"], reverse=True)
            matched_servers = server_scores[:TOP_SERVERS]
            match_result["matched_servers"] = matched_servers
            
            # Stage 2: match tools in shortlisted servers
            tool_scores = []
            
            for server_info in matched_servers:
                server = server_info["server"]
                server_score = server_info["score"]

                for tool in server["tools"]:
                    if "tool_description_embedding" not in tool:
                        continue
                    
                    tool_similarity = matcher.cosine_similarity(
                        query_tool_embedding, 
                        tool["tool_description_embedding"]
                    )
                    
                    # Combine server and tool scores (consistent with matcher.py)
                    final_score = (server_score * tool_similarity) * max(server_score, tool_similarity)
                    
                    tool_scores.append({
                        "server_name": server["name"],
                        "tool_name": tool["name"],
                        "tool_description": tool.get("description", ""),
                        "parameters": tool.get("parameter", {}),
                        "server_score": server_score,
                        "tool_score": tool_similarity,
                        "final_score": final_score
                    })
            
            tool_scores.sort(key=lambda x: x["final_score"], reverse=True)
            matched_tools = tool_scores[:TOP_TOOLS]
            match_result["matched_tools"] = matched_tools
            
            # Evaluate correctness using top-1
            if matched_tools:
                matched_tool = matched_tools[0]
                matched_server = next((s for s in matched_servers if s["server"]["name"] == matched_tool["server_name"]), None)
                
                is_correct = (
                    matched_tool["server_name"] == target_server.get("name", "") and
                    matched_tool["tool_name"] == target_tool.get("name", "")
                )
                
        except Exception as e:
            print(f"Error: {str(e)}")
            match_result["success"] = False
            match_result["error"] = str(e)
    
    # VRAM after matching
    print_memory_usage("Vector matching end")
    
    # Build final result
    result = {
        "success": success,
        "elapsed_time": elapsed_time,
        "response": response.strip(),
        "extracted_server_desc": extracted_server_desc,
        "extracted_tool_desc": extracted_tool_desc,
        "is_correct": is_correct,
        "target_server_name": target_server.get("name", ""),
        "target_tool_name": target_tool.get("name", ""),
        "target_server_description": server_description,
        "target_tool_description": tool_description,
        "matched_server": matched_server["server"]["name"] if matched_server else None,
        "matched_tool": matched_tool["tool_name"] if matched_tool else None,
        "sample_size": sample_size,
        "position_index": position_index,
        "selection_method": selection_method,
        "model_name": model_name,  # record actual model name used
    }
    
    # VRAM at test end
    print_memory_usage("Test end")
    
    return result, None


def run_grid_search(
    data_path: str,
    output_dir: Optional[str] = None,
    num_position_ratios: int = 20,
    num_sample_sizes: int = 50,
    request_interval: float = 5.0,
    model_name: str = MODEL_NAME,  # default use global
) -> None:
    # VRAM at grid search start
    print_memory_usage("Grid search start")
    """
    Run grid search evaluation.

    Args:
        data_path: Data file path
        output_dir: Output directory path
        num_position_ratios: Number of position partitions (plus one effectively)
        num_sample_sizes: Number of sample sizes
        request_interval: Interval between requests in seconds
    """
    # Output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Grid parameters
    grid_points_list = generate_grid_search_params(num_position_ratios, num_sample_sizes)
    
    # Results directory
    results_dir = os.path.join(output_dir, "grid_search_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Results list
    all_results = []
    
    # Results file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"ours_grid_search_results_{timestamp}.json")
    
    # Sampler
    sampler = ToolSampler(data_path)
    config = load_sampler_config()
    sample_threshold = config.get('sample_threshold', 10)
    # Cache for sampled data (reserved for potential reuse)
    sampled_data_cache = {}
    
    for (position_index, sample_size) in grid_points_list:
        print(f"\n=== Processing sample: {position_index} / {sample_size} ===")
        position_index = int(position_index)
        sample_size = int(sample_size)
        # 1. Initial sampling for selecting target (for evaluation and prompt construction)
        tmp_data = sampler.sample_tools(sample_size)
        # 1. Sampling strategy
        if sample_size < sample_threshold:
            print("Using random sampling")
            sampled_data = sampler.sample_tools(sample_size)
        else:
            print("Using clustered sampling")
            # Select target first to build user query
            tmp_data = sampler.sample_tools(sample_size)
            target_server, target_tool, _ = sampler.select_target_tool(tmp_data, position_index)
            server_description = target_server.get("description", "")
            tool_description = target_tool.get("description", "")
            user_prompt_template_path = os.path.join(output_dir, "ours_user_prompt_template.txt")
            user_query = generate_user_query_from_template(user_prompt_template_path, server_description, tool_description)
            sampled_data = sampler.sample_with_embedding(user_query, config)
        # 2. Select target based on sampled_data
        target_server, target_tool, _ = sampler.select_target_tool(sampled_data, position_index)
        # 3. Match/evaluate
        result, _ = test_llm_retrieval(
            sampled_data=sampled_data,
            target_server=target_server,
            target_tool=target_tool,
            sample_size=sample_size,
            position_index=position_index,
            use_random_selection=False,
            output_dir=output_dir,
            model_name=model_name  # use provided/global model name
        )
        
        # Append to results
        all_results.append(result)
        
        # Save progress
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # Print brief result
        print(f"  Result: {'✓' if result['is_correct'] else '✗'} "
              f"Elapsed: {result['elapsed_time']:.2f}s")
        print(f"  Target server: {result['target_server_name']}, target tool: {result['target_tool_name']}")
        print(f"  Extracted server description: {result['extracted_server_desc']}")
        print(f"  Extracted tool description: {result['extracted_tool_desc']}")
        print(f"  Matched server: {result['matched_server']}, matched tool: {result['matched_tool']}")
        print(f"  Model used: {result['model_name']}")
        
        # Throttle requests
        if position_index != len(grid_points_list) - 1 or sample_size != len(grid_points_list) - 1:
            print(f"  Waiting {request_interval} seconds...")
            time.sleep(request_interval)
    
    print(f"\n=== Grid search completed ===")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Results saved to: {results_file}")
    
    # Compute accuracy
    correct_count = sum(1 for result in all_results if result["is_correct"])
    accuracy = correct_count / len(all_results) if all_results else 0
    print(f"Overall accuracy: {accuracy:.2%}")

    # Log accuracy per sample size
    from collections import defaultdict
    size2acc = defaultdict(lambda: [0, 0])  # {sample_size: [correct, total]}
    for result in all_results:
        size = result.get("sample_size")
        if result.get("is_correct"):
            size2acc[size][0] += 1
        size2acc[size][1] += 1
    log_path = os.path.join(output_dir, "mistral_accuracy.log")
    for size in sorted(size2acc.keys()):
        acc = size2acc[size][0] / size2acc[size][1] if size2acc[size][1] else 0
        print(f"Accuracy for sample size {size}: {acc:.2%}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Sample size: {size}, Accuracy: {acc:.4f}, Total: {size2acc[size][1]}\n")

    # VRAM at grid search end
    print_memory_usage("Grid search end")


if __name__ == "__main__":
    data_path = ""
    run_grid_search(
        data_path=data_path,
        num_position_ratios=20,
        num_sample_sizes=50,
        request_interval=0,
        model_name=MODEL_NAME
    )