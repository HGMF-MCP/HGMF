#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from typing import List, Dict, Any, Tuple
import copy
import os
import sys
import hashlib
import pickle
import time
from functools import wraps
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.linalg import eigh
import warnings
from tqdm import tqdm

class CacheManager:
    def __init__(self, cache_dir: str = "cache", enable_disk_cache: bool = True):
        self.cache_dir = cache_dir
        self.enable_disk_cache = enable_disk_cache

        # Caches
        self.embedding_cache = {}       # text_hash -> embedding
        self.cluster_cache = {}         # cluster_params_hash -> cluster_results

        if self.enable_disk_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.embedding_cache_file = os.path.join(cache_dir, "embeddings.pkl")
        self.cluster_cache_file = os.path.join(cache_dir, "clusters.pkl")

        self._load_disk_cache()

        self.cache_stats = {
            'embedding_hits': 0,
            'embedding_misses': 0,
            'cluster_hits': 0,
            'cluster_misses': 0
        }
    
    def _get_hash(self, obj) -> str:
        if isinstance(obj, str):
            return hashlib.md5(obj.encode('utf-8')).hexdigest()
        elif isinstance(obj, (list, tuple)):
            return hashlib.md5(str(obj).encode('utf-8')).hexdigest()
        elif isinstance(obj, dict):
            return hashlib.md5(json.dumps(obj, sort_keys=True).encode('utf-8')).hexdigest()
        elif isinstance(obj, np.ndarray):
            return hashlib.md5(obj.tobytes()).hexdigest()
        else:
            return hashlib.md5(str(obj).encode('utf-8')).hexdigest()
    
    def _load_disk_cache(self):
        if not self.enable_disk_cache:
            return
        
        import builtins
        
        cache_files = [
            (self.embedding_cache_file, self.embedding_cache),
            (self.cluster_cache_file, self.cluster_cache)
        ]
        
        for cache_file, cache_dict in cache_files:
            if os.path.exists(cache_file):
                try:
                    with builtins.open(cache_file, 'rb') as f:
                        cache_dict.update(pickle.load(f))
                    print(f"Loaded cache file: {cache_file}, entries: {len(cache_dict)}")
                except Exception as e:
                    print(f"Failed to load cache file {cache_file}: {e}")
    
    def _save_disk_cache(self):
        if not self.enable_disk_cache:
            return
        
        import builtins
        
        cache_files = [
            (self.embedding_cache_file, self.embedding_cache),
            (self.cluster_cache_file, self.cluster_cache)
        ]
        
        for cache_file, cache_dict in cache_files:
            try:
                with builtins.open(cache_file, 'wb') as f:
                    pickle.dump(cache_dict, f)
            except Exception as e:
                print(f"Failed to save cache file {cache_file}: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        text_hash = self._get_hash(text)
        if text_hash in self.embedding_cache:
            self.cache_stats['embedding_hits'] += 1
            return self.embedding_cache[text_hash]
        
        self.cache_stats['embedding_misses'] += 1
        embedding = self._compute_embedding_original(text)
        self.embedding_cache[text_hash] = embedding
        return embedding
    
    def _compute_embedding_original(self, text: str) -> np.ndarray:
        global _embedding_model
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        return _embedding_model.encode(text, show_progress_bar=False)
    
    def get_cluster_results(self, cluster_params: dict, cluster_func, *args) -> tuple:
        params_hash = self._get_hash(cluster_params)
        if params_hash in self.cluster_cache:
            self.cache_stats['cluster_hits'] += 1
            return self.cluster_cache[params_hash]
        
        self.cache_stats['cluster_misses'] += 1
        cluster_results = cluster_func(*args)
        self.cluster_cache[params_hash] = cluster_results
        return cluster_results
    
    def clear_cache(self, cache_type: str = 'all'):
        if cache_type == 'all' or cache_type == 'embedding':
            self.embedding_cache.clear()
        if cache_type == 'all' or cache_type == 'cluster':
            self.cluster_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        total_hits = sum(v for k, v in self.cache_stats.items() if 'hits' in k)
        total_misses = sum(v for k, v in self.cache_stats.items() if 'misses' in k)
        total_requests = total_hits + total_misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        return {
            **self.cache_stats,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_sizes': {
                'embedding': len(self.embedding_cache),
                'cluster': len(self.cluster_cache)
            }
        }
    
    def save_cache(self):
        self._save_disk_cache()
    
    def __del__(self):
        try:
            self._save_disk_cache()
        except:
            pass


_cache_manager = None

def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cache_embedding(func):
    @wraps(func)
    def wrapper(text_or_list):
        cache_mgr = get_cache_manager()
        if isinstance(text_or_list, str):
            return cache_mgr.get_embedding(text_or_list)
        elif isinstance(text_or_list, list):
            results = []
            for text in text_or_list:
                results.append(cache_mgr.get_embedding(text))
            return np.array(results)
        else:
            return func(text_or_list)
    return wrapper

def _compute_embedding_without_cache(text_or_list):
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return _embedding_model.encode(text_or_list, show_progress_bar=False)


DEFAULT_CONFIG = {
    'sample_threshold': ,      # Threshold for using clustering
    'n_clusters': None,          # If None, auto by sqrt(N)
    'sample_num': ,            # Max samples per selected group
    'topk_cluster': ,           # Top-k clusters for selection
    'lambda_inter': ,        # Mean regularization weight (towards ETF)
    'beta_intra':,          # Covariance trace penalty weight
    'w_balance': ,           # Covariance directional penalty weight
    'max_iter': ,             # Max EM iterations
    'tol': ,                 # Convergence tolerance on log-likelihood
    'reg_covar':            # Base covariance regularization
}

def load_sampler_config(config_path: str = 'config.json') -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        return config
    else:
        return DEFAULT_CONFIG.copy()


EMBEDDING_MODEL_PATH = r""
_embedding_model = None

@cache_embedding
def get_embedding(text_or_list):
    return _compute_embedding_without_cache(text_or_list)

def generate_etf_vectors(K, d):
    if K == 1:
        return np.ones((1, d)) / np.sqrt(d)
    U = np.random.randn(d, K)
    U, _ = np.linalg.qr(U)
    H = np.eye(K) - np.ones((K, K)) / K
    M = (np.sqrt(K / (K - 1)) * (U @ H)).T
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    M = M / norms
    return M

def safe_log(x, eps=1e-8):
    return np.log(np.clip(x, eps, None))

def safe_inv(cov, reg_covar=1e-6):
    try:
        cov_reg = cov + reg_covar * np.eye(cov.shape[0])
        return np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov + reg_covar * np.eye(cov.shape[0]))

class ImprovedGMM:
    def __init__(self, n_components, config=None):
        self.n_components = n_components
        self.config = config or DEFAULT_CONFIG

        self.lambda_inter = self.config.get('lambda_inter', DEFAULT_CONFIG['lambda_inter'])
        self.beta_intra = self.config.get('beta_intra', DEFAULT_CONFIG['beta_intra'])
        self.w_balance = self.config.get('w_balance', DEFAULT_CONFIG['w_balance'])
        self.max_iter = self.config.get('max_iter', DEFAULT_CONFIG['max_iter'])
        self.tol = self.config.get('tol', DEFAULT_CONFIG['tol'])
        self.reg_covar = self.config.get('reg_covar', DEFAULT_CONFIG['reg_covar'])
        
        # Model parameters
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.etf_vectors_ = None
        self.converged_ = False
        self.n_iter_ = 0
        
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.etf_vectors_ = generate_etf_vectors(self.n_components, n_features)
        self.means_ = self.etf_vectors_.copy()
        self.covariances_ = np.array([np.eye(n_features) * 0.1 for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components
        
    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            try:
                diff = X - self.means_[k]
                cov_inv = safe_inv(self.covariances_[k], self.reg_covar)
                mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
                log_det = np.linalg.slogdet(self.covariances_[k])[1]
                log_prob = -0.5 * (mahal_dist + log_det + X.shape[1] * np.log(2 * np.pi))
                responsibilities[:, k] = self.weights_[k] * np.exp(log_prob)
            except (np.linalg.LinAlgError, RuntimeWarning):
                distances = np.linalg.norm(X - self.means_[k], axis=1)
                responsibilities[:, k] = self.weights_[k] * np.exp(-distances / 2)
        
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-8, row_sums, 1.0)
        responsibilities = responsibilities / row_sums
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        Nk = np.maximum(Nk, 1e-6)
        self.weights_ = Nk / n_samples

        for k in range(self.n_components):
            weighted_sum = np.sum(responsibilities[:, k:k+1] * X, axis=0)
            numerator = weighted_sum + self.lambda_inter * self.etf_vectors_[k]
            denominator = Nk[k] + self.lambda_inter
            self.means_[k] = numerator / denominator

        for k in range(self.n_components):
            try:
                diff = X - self.means_[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                cov = (weighted_diff.T @ diff) / Nk[k]
                cov = (cov + cov.T) / 2

                base_reg = max(self.reg_covar, 1e-4)
                identity_reg = base_reg * np.eye(n_features)

                trace_penalty = self.beta_intra * np.eye(n_features)

                cosine_penalty = np.zeros_like(cov)
                if Nk[k] > 1:
                    high_resp_mask = responsibilities[:, k] > 0.1
                    if np.any(high_resp_mask):
                        mean_norm = np.linalg.norm(self.means_[k])
                        if mean_norm > 1e-8:
                            cosine_penalty = self.w_balance / Nk[k] * np.outer(
                                self.means_[k], self.means_[k]
                            ) / (mean_norm ** 2)
                
                self.covariances_[k] = cov + identity_reg + trace_penalty + cosine_penalty

                try:
                    eigenvals, eigenvecs = eigh(self.covariances_[k])
                    min_eigenval = max(base_reg, np.max(eigenvals) * 1e-6)
                    eigenvals = np.maximum(eigenvals, min_eigenval)
                    self.covariances_[k] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                except np.linalg.LinAlgError:
                    diag_cov = np.diag(np.diag(cov))
                    self.covariances_[k] = diag_cov + identity_reg
            except Exception as e:
                print(f"Warning: Failed to update covariance matrix for cluster {k}: {e}, using identity.")
                self.covariances_[k] = np.eye(n_features) * 0.1
    
    def _compute_log_likelihood(self, X, responsibilities):
        log_likelihood = 0
        for k in range(self.n_components):
            if self.weights_[k] > 1e-8:
                diff = X - self.means_[k]
                cov_inv = safe_inv(self.covariances_[k], self.reg_covar)
                mahal_dist = np.sum(diff @ cov_inv * diff, axis=1)
                log_det = np.linalg.slogdet(self.covariances_[k])[1]
                log_prob = -0.5 * (mahal_dist + log_det + X.shape[1] * np.log(2 * np.pi))
                log_likelihood += np.sum(
                    responsibilities[:, k] * (safe_log(self.weights_[k]) + log_prob)
                )
        return log_likelihood
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            warnings.warn("The number of samples is less than the number of clusters.")
        print(f"Begin training GMM (samples: {X.shape[0]}, clusters: {self.n_components})")
        self._initialize_parameters(X)
        prev_log_likelihood = -np.inf

        pbar = tqdm(range(self.max_iter), desc="GMM Training", leave=False)
        for i in pbar:
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood = self._compute_log_likelihood(X, responsibilities)

            pbar.set_postfix({
                'log_likelihood': f'{log_likelihood:.2f}',
                'converged': self.converged_
            })
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                pbar.set_postfix({
                    'log_likelihood': f'{log_likelihood:.2f}',
                    'converged': True
                })
                break
            prev_log_likelihood = log_likelihood
            self.n_iter_ = i + 1

            if not np.isfinite(log_likelihood):
                warnings.warn("Detected unstable values; stopped early.")
                break
        
        pbar.close()
        print(f"Done training GMM (iterations: {self.n_iter_}, converged: {self.converged_})")
        return self
    
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        return self._e_step(X)

def cluster_embeddings_improved(embeddings, n_clusters, config=None):
    if config is None:
        config = DEFAULT_CONFIG

    embeddings = np.asarray(embeddings, dtype=np.float64)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    cache_mgr = get_cache_manager()
    cluster_params = {
        'n_clusters': n_clusters,
        'config': config,
        'embeddings_hash': cache_mgr._get_hash(embeddings_norm)
    }
    
    def _cluster_func():
        gmm = ImprovedGMM(n_components=n_clusters, config=config)
        gmm.fit(embeddings_norm)
        labels = gmm.predict(embeddings_norm)
        responsibilities = gmm.predict_proba(embeddings_norm)
        return gmm.means_, labels, responsibilities, gmm
    
    return cache_mgr.get_cluster_results(cluster_params, _cluster_func)

def find_topk_clusters(user_emb, cluster_centers, k, gmm=None):
    if gmm is not None:
        user_emb = np.array(user_emb, dtype=np.float64).reshape(1, -1)
        user_emb = user_emb / (np.linalg.norm(user_emb, axis=1, keepdims=True) + 1e-8)
        probs = gmm.predict_proba(user_emb)[0]
        return np.argsort(probs)[-k:][::-1]
    else:
        dists = np.linalg.norm(cluster_centers - user_emb, axis=1)
        return np.argsort(dists)[:k]

class ToolSampler:
    """
    Sample a specified number of tools from the dataset and select a target tool.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.servers_data = None
        self.all_tools = []  # (server_index, tool_index, tool_data)
        self.load_data()
        
    def load_data(self) -> None:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.servers_data = json.load(f)
            for server_idx, server in enumerate(self.servers_data):
                if "tools" in server and server["tools"]:
                    for tool_idx, tool in enumerate(server["tools"]):
                        self.all_tools.append((server_idx, tool_idx, tool))
            print(f"Count: {len(self.servers_data)} servers, {len(self.all_tools)} tools.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def sample_tools(self, n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            raise ValueError
        if not self.servers_data:
            raise ValueError('servers_data is empty or None')
        if n >= len(self.all_tools):
            return copy.deepcopy(self.servers_data)

        sampled_tool_indices = random.sample(range(len(self.all_tools)), n)
        sampled_tools = [self.all_tools[i] for i in sampled_tool_indices]
        server_tool_map = {}
        for server_idx, tool_idx, _ in sampled_tools:
            if server_idx not in server_tool_map:
                server_tool_map[server_idx] = []
            server_tool_map[server_idx].append(tool_idx)
        result = []
        for server_idx, tool_indices in server_tool_map.items():
            if not isinstance(self.servers_data, list) or server_idx >= len(self.servers_data):
                continue
            server_copy = {k: v for k, v in self.servers_data[server_idx].items() if k != "tools"}
            server_copy["tools"] = [
                copy.deepcopy(self.servers_data[server_idx]["tools"][tool_idx])
                for tool_idx in tool_indices
            ]
            result.append(server_copy)
        return result
    
    def select_target_tool(self, sampled_data: List[Dict[str, Any]], position_index: float = 0.0) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        if not sampled_data:
            raise ValueError
        all_sampled_tools = []
        for server in sampled_data:
            if "tools" in server and server["tools"]:
                for tool in server["tools"]:
                    all_sampled_tools.append((server, tool))
        if not all_sampled_tools:
            raise ValueError

        if isinstance(position_index, float) and 0 <= position_index <= 1:
            target_index = int(position_index * (len(all_sampled_tools) - 1))
        else:
            target_index = max(0, min(len(all_sampled_tools)-1, int(position_index)))
        target_server, target_tool = all_sampled_tools[target_index]
        tool_fullname = f"{target_server['name']}::{target_tool['name']}"
        return target_server, target_tool, tool_fullname

    def select_target_tool_random(self, sampled_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Select a tool randomly.
        Returns:
            (target_server, target_tool, tool_fullname)
        """
        if not sampled_data:
            raise ValueError
        all_sampled_tools = []
        for server in sampled_data:
            if "tools" in server and server["tools"]:
                for tool in server["tools"]:
                    all_sampled_tools.append((server, tool))
        if not all_sampled_tools:
            raise ValueError
        target_server, target_tool = random.choice(all_sampled_tools)
        tool_fullname = f"{target_server['name']}::{target_tool['name']}"
        return target_server, target_tool, tool_fullname

    def sample_with_embedding(self, user_query: str, config: dict = None):
        if config is None:
            config = load_sampler_config()
        
        print("Begin clustered sampling...")
        start_time = time.time()
    
        print("Computing embedding for user query...")
        user_emb = get_embedding(user_query)
        
        print(f"Processing embeddings for {len(self.servers_data)} servers...")
        server_embs = []
        for i, server in enumerate(tqdm(self.servers_data, desc="Server Embeddings", leave=False)):
            emb = server.get('embedding', None)
            if emb is not None:
                server_embs.append(emb)
            else:
                server_embs.append(get_embedding(server['name']))
        
        server_embs = np.array(server_embs)
        n_servers = len(server_embs)
        
        if n_servers > config['sample_threshold']:
            n_clusters = config.get('n_clusters')
            if n_clusters is None or n_clusters <= 0:
                n_clusters = max(2, int(np.sqrt(n_servers)))
            
            print(f"Clustering {n_servers} servers into {n_clusters} clusters...")
            cluster_centers, labels, responsibilities, gmm = cluster_embeddings_improved(server_embs, n_clusters, config)
            topk_idx = find_topk_clusters(user_emb, cluster_centers, config['topk_cluster'], gmm)
            selected_servers = [self.servers_data[i] for i in range(n_servers) if labels[i] in topk_idx]
            print(f"Selected {len(selected_servers)} relevant servers")
        else:
            print(f"Number of servers ({n_servers}) <= threshold ({config['sample_threshold']}), using all servers")
            selected_servers = self.servers_data
        
        print("Processing tools from selected servers...")
        result = []
        total_tools_processed = 0
        
        for server_idx, server in enumerate(tqdm(selected_servers, desc="Processing Server Tools", leave=False)):
            tools = server.get('tools', [])
            if not tools:
                continue

            tool_embs = []
            for tool in tools:
                emb = tool.get('embedding', None)
                if emb is not None:
                    tool_embs.append(emb)
                else:
                    tool_embs.append(get_embedding(tool['name']))
            tool_embs = np.array(tool_embs)
            n_tools = len(tool_embs)
            total_tools_processed += n_tools
            
            if n_tools > config['sample_threshold']:
                n_clusters = config.get('n_clusters')
                if n_clusters is None or n_clusters <= 0:
                    n_clusters = max(2, int(np.sqrt(n_tools)))
                cluster_centers, labels, responsibilities, gmm = cluster_embeddings_improved(tool_embs, n_clusters, config)
                topk_idx = find_topk_clusters(user_emb, cluster_centers, config['topk_cluster'], gmm)
                selected_tools = [tools[i] for i in range(n_tools) if labels[i] in topk_idx]
            else:
                selected_tools = tools
            
            if len(selected_tools) > config['sample_num']:
                sampled_tools = random.sample(selected_tools, config['sample_num'])
            else:
                sampled_tools = selected_tools
            server_copy = {k: v for k, v in server.items() if k != 'tools'}
            server_copy['tools'] = sampled_tools
            result.append(server_copy)
        
        total_sampled_tools = sum(len(server.get('tools', [])) for server in result)
        elapsed_time = time.time() - start_time
        
        print("Sampling completed!")
        print(f"Processed {total_tools_processed} tools")
        print(f"Sampled {total_sampled_tools} tools")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        
        return result