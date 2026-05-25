#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple


def generate_grid_search_params(
    num_position_ratios: int = 20,
    num_sample_sizes: int = 50,
    total_tools: int = 2797
) -> List[List[int]]:
    sample_sizes = [41]
    position_ratios = np.linspace(0, 1, num_position_ratios+1).tolist()
    position_indexes = []
    for sample_size in sample_sizes:
        for position_ratio in position_ratios:
            target = [int((sample_size - 1) * position_ratio), sample_size]
            if target not in position_indexes:
                position_indexes.append(target)
    return position_indexes

def generate_user_query_from_template(template_path: str, server_description: str, tool_description: str) -> str:
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    user_query = template.replace("{server_description}", server_description)
    user_query = user_query.replace("{tool_description}", tool_description)
    return user_query


if __name__ == "__main__":
    grid_points_list = generate_grid_search_params(num_position_ratios=2, num_sample_sizes=3)
    print(grid_points_list)