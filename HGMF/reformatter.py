#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import List, Dict, Any, Tuple
import copy

from sampler import ToolSampler


def extract_parameter_type(param_desc: str) -> Tuple[str, str]:
    if not param_desc or not param_desc.startswith("("):
        return "string", param_desc
    
    try:
        type_end = param_desc.find(")")
        if type_end == -1:
            return "string", param_desc
        
        param_type = param_desc[1:type_end].strip().lower()
        description = param_desc[type_end+1:].strip()
        
        type_mapping = {
            "string": "string",
            "str": "string",
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "object": "object",
            "dict": "object"
        }
        
        mapped_type = type_mapping.get(param_type, "string")
        return mapped_type, description
    except Exception:
        return "string", param_desc


def format_parameters(parameters: Dict[str, str]) -> Dict[str, Any]:
    if not parameters:
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": []
        }
    
    properties = {}
    required = []
    
    for param_name, param_desc in parameters.items():
        param_type, param_description = extract_parameter_type(param_desc)
        
        properties[param_name] = {
            "description": param_description,
            "type": param_type
        }
        
        if not 'optional' in param_description.lower():
            required.append(param_name)
    
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
        "type": "object"
    }


def format_tools_as_functions(servers_data: List[Dict[str, Any]]) -> str:
    """
    Format tool descriptions 
    """
    formatted_functions = []
    
    for server in servers_data:
        server_name = server.get("name", "unknown_server")
        server_description = server.get("description", "")
        
        if "tools" not in server or not server["tools"]:
            continue
        
        for tool in server["tools"]:
            tool_name = tool.get("name", "unknown_tool")
            tool_description = tool.get("description", "")
            
            parameters = {}
            if "parameter" in tool and tool["parameter"]:
                parameters = tool["parameter"]
            formatted_parameters = format_parameters(parameters)
            
            function_obj = {
                "description": tool_description,
                "name": f"mcp_{server_name.lower()}_{tool_name.lower()}",
                "parameters": formatted_parameters
            }
            formatted_function = f'<function>{json.dumps(function_obj, ensure_ascii=False)}</function>'
            formatted_functions.append(formatted_function)
    
    return "\n".join(formatted_functions)
