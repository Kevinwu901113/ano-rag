import networkx as nx
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import math
import time
from datetime import datetime, timedelta
from .relation_extractor import RelationExtractor
from config import config

try:
    from .graph_quality import compute_metrics
except Exception:  # pragma: no cover - optional dependency
    compute_metrics = None  # type: ignore
class GraphBuilder:
    """Builds a knowledge graph from atomic notes and relations."""
    def __init__(self, llm=None):
        # 使用统一的RelationExtractor，支持增强功能
        self.relation_extractor = RelationExtractor(local_llm=llm)
        
        # 原子笔记粒度的图构建配置
        self.atomic_note_config = {
            'enable_atomic_nodes': config.get('graph.enable_atomic_nodes', True),
            'enable_entity_nodes': config.get('graph.enable_entity_nodes', True),
            'enable_structural_edges': config.get('graph.enable_structural_edges', True),
            'max_edges_per_node': config.get('graph.max_edges_per_node', 50),
            'hub_degree_threshold': config.get('graph.hub_degree_threshold', 20)
        }
        
        # 时间衰减权重配置
        self.time_decay_config = {
            'enabled': config.get('graph.time_decay.enabled', True),
            'lambda': config.get('graph.time_decay.lambda', 0.1),
            'reference_time': config.get('graph.time_decay.reference_time', 'current'),
            'time_unit': config.get('graph.time_decay.time_unit', 'days')
        }
        
        # 反hub惩罚配置
        self.hub_penalty_config = {
            'enabled': config.get('graph.hub_penalty.enabled', True),
            'strength': config.get('graph.hub_penalty.strength', 0.5),
            'threshold': config.get('graph.hub_penalty.threshold', 10),
            'method': config.get('graph.hub_penalty.method', 'log_normalization')
        }
        
        # metapath配置
        self.metapath_config = {
            'enabled': config.get('graph.metapaths.enabled', True),
            'whitelist': config.get('graph.metapaths.whitelist', []),
            'path_preferences': config.get('graph.metapaths.path_preferences', {}),
            'max_path_length': config.get('graph.metapaths.max_path_length', 3)
        }

    def build_graph(self, atomic_notes: List[Dict[str, Any]], embeddings=None) -> nx.Graph:
        """Create a graph from notes and optional embeddings with atomic note granularity."""
        logger.info(f"Building graph with {len(atomic_notes)} notes")
        
        # 输出关键参数的实际值
        logger.info(f"Graph building parameters:")
        logger.info(f"  Time decay: enabled={self.time_decay_config['enabled']}, lambda={self.time_decay_config['lambda']}")
        logger.info(f"  Hub penalty: enabled={self.hub_penalty_config['enabled']}, threshold={self.hub_penalty_config['threshold']}, strength={self.hub_penalty_config['strength']}")
        logger.info(f"  Metapath filtering: enabled={self.metapath_config['enabled']}, whitelist={self.metapath_config['whitelist']}")
        
        G = nx.Graph()
        
        # 添加原子笔记节点，包含详细属性
        for note in tqdm(atomic_notes, desc="Adding atomic note nodes"):
            node_id = note.get("note_id")
            if not node_id:
                continue
            
            # 构建原子笔记节点属性
            node_attrs = self._build_atomic_note_attributes(note)
            G.add_node(node_id, **node_attrs)
            
            # 如果启用实体节点，为每个实体创建节点
            if self.atomic_note_config['enable_entity_nodes']:
                self._add_entity_nodes(G, note, node_id)
        
        # 提取关系（包括新增的原子笔记粒度关系）
        relations = self.relation_extractor.extract_all_relations(atomic_notes, embeddings)
        
        # 添加原子笔记间的关系
        for rel in tqdm(relations, desc="Adding note relations"):
            src = rel.get("source_id")
            tgt = rel.get("target_id")
            weight = rel.get("weight", 1.0)
            rtype = rel.get("relation_type")
            if src and tgt and G.has_node(src) and G.has_node(tgt):
                G.add_edge(src, tgt, weight=weight, relation_type=rtype, **rel.get("metadata", {}))
        
        # 添加结构化边（如果启用）
        if self.atomic_note_config['enable_structural_edges']:
            self._add_structural_edges(G, atomic_notes)
        
        # 应用时间衰减权重
        if self.time_decay_config['enabled']:
            G = self._apply_time_decay_weights(G)
        
        # 应用边去噪和hub处理
        G = self._apply_edge_denoising(G)
        G = self._apply_hub_normalization(G)
        
        # 应用增强的反hub惩罚
        if self.hub_penalty_config['enabled']:
            G = self._apply_enhanced_hub_penalty(G)
        
        # 应用metapath过滤
        if self.metapath_config['enabled']:
            G = self._apply_metapath_filtering(G)
        
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _build_atomic_note_attributes(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """构建原子笔记节点的属性字典"""
        attrs = {
            # 基础属性
            'node_type': 'atomic_note',
            'content': note.get('content', ''),
            'summary': note.get('summary', ''),
            'title': note.get('title', ''),
            'original_text': note.get('original_text', ''),
            
            # 实体和概念
            'entities': note.get('entities', []),
            'normalized_entities': note.get('normalized_entities', []),
            'concepts': note.get('concepts', []),
            'keywords': note.get('keywords', []),
            
            # 谓词和关系
            'predicate': note.get('predicate', ''),
            'normalized_predicates': note.get('normalized_predicates', []),
            'relations': note.get('relations', []),
            
            # 时间和位置信息
            'timestamp': note.get('timestamp', ''),
            'sentence_index': note.get('sentence_index', 0),
            'char_span': note.get('char_span'),
            'group_index': note.get('group_index', 0),
            
            # 重要性和类型
            'importance_score': note.get('importance_score', 0.5),
            'note_type': note.get('note_type', 'fact'),
            
            # 结构信息
            'chunk_index': note.get('chunk_index', 0),
            'paragraph_idxs': note.get('paragraph_idxs', []),
            'source_info': note.get('source_info', {}),
            'length': note.get('length', 0)
        }
        return attrs
    
    def _add_entity_nodes(self, G: nx.Graph, note: Dict[str, Any], note_id: str):
        """为原子笔记中的实体创建节点并建立连接"""
        entities = note.get('entities', [])
        normalized_entities = note.get('normalized_entities', [])
        
        # 使用归一化实体，如果没有则使用原始实体
        entity_list = normalized_entities if normalized_entities else entities
        
        for entity in entity_list:
            if not entity or not isinstance(entity, str):
                continue
                
            entity_id = f"entity_{entity.lower().replace(' ', '_')}"
            
            # 添加实体节点（如果不存在）
            if not G.has_node(entity_id):
                G.add_node(entity_id, 
                          node_type='entity',
                          entity_name=entity,
                          mention_count=1,
                          connected_notes=[note_id])
            else:
                # 更新实体节点的统计信息
                G.nodes[entity_id]['mention_count'] += 1
                if note_id not in G.nodes[entity_id]['connected_notes']:
                    G.nodes[entity_id]['connected_notes'].append(note_id)
            
            # 添加笔记到实体的边
            G.add_edge(note_id, entity_id, 
                       relation_type='contains_entity',
                       weight=0.8)
    
    def _add_structural_edges(self, G: nx.Graph, atomic_notes: List[Dict[str, Any]]):
        """添加结构化边，包括层级信息和文档结构"""
        # 按文档和段落组织笔记
        doc_structure = defaultdict(lambda: defaultdict(list))
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            if not note_id:
                continue
                
            source_info = note.get('source_info', {})
            doc_id = source_info.get('file_path', 'unknown_doc')
            chunk_index = note.get('chunk_index', 0)
            
            doc_structure[doc_id][chunk_index].append(note)
        
        # 为每个文档添加结构边
        for doc_id, chunks in doc_structure.items():
            # 添加文档节点
            doc_node_id = f"doc_{doc_id.replace('/', '_').replace('.', '_')}"
            if not G.has_node(doc_node_id):
                G.add_node(doc_node_id, 
                          node_type='document',
                          doc_path=doc_id,
                          chunk_count=len(chunks))
            
            # 连接文档到其包含的笔记
            for chunk_index, notes_in_chunk in chunks.items():
                # 添加段落节点
                chunk_node_id = f"chunk_{doc_id}_{chunk_index}".replace('/', '_').replace('.', '_')
                if not G.has_node(chunk_node_id):
                    G.add_node(chunk_node_id,
                              node_type='chunk',
                              doc_id=doc_id,
                              chunk_index=chunk_index,
                              note_count=len(notes_in_chunk))
                
                # 文档到段落的边
                G.add_edge(doc_node_id, chunk_node_id,
                          relation_type='contains_chunk',
                          weight=1.0)
                
                # 段落到笔记的边
                for note in notes_in_chunk:
                    note_id = note.get('note_id')
                    if note_id and G.has_node(note_id):
                        G.add_edge(chunk_node_id, note_id,
                                  relation_type='contains_note',
                                  weight=0.9)
                        
                        # 同段落内相邻笔记的边
                        for other_note in notes_in_chunk:
                            other_note_id = other_note.get('note_id')
                            if (other_note_id and other_note_id != note_id and 
                                G.has_node(other_note_id)):
                                sentence_diff = abs(note.get('sentence_index', 0) - 
                                                   other_note.get('sentence_index', 0))
                                if sentence_diff <= 2:  # 相邻句子
                                    weight = 0.7 - (sentence_diff * 0.1)
                                    G.add_edge(note_id, other_note_id,
                                              relation_type='adjacent_sentence',
                                              weight=weight,
                                              sentence_distance=sentence_diff)
    
    def _apply_edge_denoising(self, G: nx.Graph) -> nx.Graph:
        """应用每节点每边型Top-m和互为top-k规则去噪"""
        max_edges = self.atomic_note_config['max_edges_per_node']
        top_m_per_type = self.atomic_note_config.get('top_m_per_edge_type', 5)
        mutual_top_k = self.atomic_note_config.get('mutual_top_k', 3)
        
        # 收集所有边信息
        all_edges = []
        for u, v, data in G.edges(data=True):
            all_edges.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0.0),
                'relation_type': data.get('relation_type', 'unknown'),
                'data': data
            })
        
        # 按节点和边类型组织关系
        node_edge_relations = defaultdict(lambda: defaultdict(list))
        
        for edge in all_edges:
            source_id = edge['source']
            target_id = edge['target']
            relation_type = edge['relation_type']
            weight = edge['weight']
            
            # 为源节点记录出边
            node_edge_relations[source_id][relation_type].append({
                'edge': edge,
                'target': target_id,
                'weight': weight,
                'direction': 'out'
            })
            
            # 为目标节点记录入边
            node_edge_relations[target_id][relation_type].append({
                'edge': edge,
                'target': source_id,
                'weight': weight,
                'direction': 'in'
            })
        
        # 应用每节点每边型Top-m规则
        filtered_edges = set()
        
        for node_id, edge_types in node_edge_relations.items():
            for edge_type, edges in edge_types.items():
                # 按权重排序，保留Top-m
                edges.sort(key=lambda x: x['weight'], reverse=True)
                top_edges = edges[:top_m_per_type]
                
                for edge_info in top_edges:
                    edge = edge_info['edge']
                    edge_key = (edge['source'], edge['target'])
                    filtered_edges.add(edge_key)
        
        # 应用互为top-k规则
        mutual_edges = set()
        
        for edge in all_edges:
            source_id = edge['source']
            target_id = edge['target']
            relation_type = edge['relation_type']
            
            # 检查是否在源节点的top-k中
            source_edges = node_edge_relations[source_id][relation_type]
            source_edges.sort(key=lambda x: x['weight'], reverse=True)
            source_top_k = [e['target'] for e in source_edges[:mutual_top_k] if e['direction'] == 'out']
            
            # 检查是否在目标节点的top-k中
            target_edges = node_edge_relations[target_id][relation_type]
            target_edges.sort(key=lambda x: x['weight'], reverse=True)
            target_top_k = [e['target'] for e in target_edges[:mutual_top_k] if e['direction'] == 'in']
            
            # 如果互为top-k，保留关系
            if target_id in source_top_k and source_id in target_top_k:
                edge_key = (source_id, target_id)
                mutual_edges.add(edge_key)
        
        # 删除不符合条件的边
        edges_to_remove = []
        for u, v in G.edges():
            edge_key = (u, v)
            if edge_key not in filtered_edges and edge_key not in mutual_edges:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        
        # 添加去噪标记到保留的边
        for u, v, data in G.edges(data=True):
            edge_key = (u, v)
            data['denoising_applied'] = True
            data['kept_by_top_m'] = edge_key in filtered_edges
            data['kept_by_mutual_top_k'] = edge_key in mutual_edges
        
        logger.info(f"Edge denoising: {len(all_edges)} -> {G.number_of_edges()} edges")
        return G
    
    def _apply_hub_normalization(self, G: nx.Graph) -> nx.Graph:
        """对高频实体进行反hub处理（度归一化）"""
        if G.number_of_nodes() == 0:
            return G
        
        # 配置参数
        hub_threshold = self.atomic_note_config.get('hub_degree_threshold', 20)
        normalization_factor = self.atomic_note_config.get('hub_normalization_factor', 0.5)
        
        # 计算节点度数
        node_degrees = dict(G.degree())
        
        # 识别hub节点（度数超过阈值的节点）
        hub_nodes = [node for node, degree in node_degrees.items() if degree >= hub_threshold]
        
        if not hub_nodes:
            logger.info("No hub nodes found for normalization")
            return G
        
        logger.info(f"Found {len(hub_nodes)} hub nodes for normalization")
        
        # 对每个hub节点进行度归一化处理
        for hub_node in hub_nodes:
            hub_degree = node_degrees[hub_node]
            
            # 计算归一化因子
            # 使用对数缩放来减少hub节点的影响
            import math
            degree_penalty = math.log(hub_degree) / math.log(hub_threshold) if hub_degree > hub_threshold else 1.0
            normalization = normalization_factor / degree_penalty
            
            # 获取与hub节点相连的所有边
            hub_edges = list(G.edges(hub_node, data=True))
            
            # 对每条边应用归一化
            for u, v, data in hub_edges:
                original_weight = data.get('weight', 1.0)
                
                # 应用度归一化
                normalized_weight = original_weight * normalization
                
                # 更新边权重
                data['weight'] = normalized_weight
                data['original_weight'] = original_weight
                data['hub_normalized'] = True
                data['hub_node'] = hub_node
                data['hub_degree'] = hub_degree
                data['normalization_factor'] = normalization
            
            # 为hub节点添加标记
            if G.has_node(hub_node):
                node_data = G.nodes[hub_node]
                node_data['is_hub'] = True
                node_data['original_degree'] = hub_degree
                node_data['normalization_applied'] = True
        
        # 统计处理结果
        normalized_edges = sum(1 for _, _, data in G.edges(data=True) if data.get('hub_normalized', False))
        
        logger.info(f"Hub normalization applied to {len(hub_nodes)} nodes, {normalized_edges} edges normalized")
        return G
    
    def _apply_time_decay_weights(self, G: nx.Graph) -> nx.Graph:
        """应用时间衰减权重（老化因子）"""
        if G.number_of_edges() == 0:
            return G
        
        lambda_decay = self.time_decay_config['lambda']
        reference_time = self.time_decay_config['reference_time']
        time_unit = self.time_decay_config['time_unit']
        
        # 确定参考时间
        if reference_time == 'current':
            ref_time = datetime.now()
        else:
            try:
                ref_time = datetime.fromisoformat(reference_time)
            except (ValueError, TypeError):
                logger.warning(f"Invalid reference_time format: {reference_time}, using current time")
                ref_time = datetime.now()
        
        # 时间单位转换为秒的映射
        time_unit_seconds = {
            'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400, 'weeks': 604800
        }
        unit_multiplier = time_unit_seconds.get(time_unit, 86400)
        
        edges_processed = 0
        for u, v, data in G.edges(data=True):
            edge_time = None
            if 'timestamp' in data:
                try:
                    edge_time = datetime.fromisoformat(str(data['timestamp']))
                except (ValueError, TypeError):
                    pass
            
            if edge_time is None:
                for node_id in [u, v]:
                    if G.has_node(node_id):
                        node_data = G.nodes[node_id]
                        if 'timestamp' in node_data:
                            try:
                                edge_time = datetime.fromisoformat(str(node_data['timestamp']))
                                break
                            except (ValueError, TypeError):
                                continue
            
            if edge_time is None:
                data['time_decay_applied'] = False
                data['time_decay_factor'] = 1.0
                continue
            
            time_diff = ref_time - edge_time
            time_diff_units = time_diff.total_seconds() / unit_multiplier
            decay_factor = math.exp(-lambda_decay * max(0, time_diff_units))
            
            original_weight = data.get('weight', 1.0)
            data['weight'] = original_weight * decay_factor
            data['original_weight'] = original_weight
            data['time_decay_applied'] = True
            data['time_decay_factor'] = decay_factor
            edges_processed += 1
        
        logger.info(f"Time decay applied to {edges_processed} edges with lambda={lambda_decay}")
        return G
    
    def _apply_enhanced_hub_penalty(self, G: nx.Graph) -> nx.Graph:
        """应用增强的反hub惩罚（度归一化）"""
        if G.number_of_nodes() == 0:
            return G
        
        strength = self.hub_penalty_config['strength']
        threshold = self.hub_penalty_config['threshold']
        method = self.hub_penalty_config['method']
        
        node_degrees = dict(G.degree())
        hub_nodes = [node for node, degree in node_degrees.items() if degree >= threshold]
        
        if not hub_nodes:
            logger.info("No hub nodes found for enhanced penalty")
            return G
        
        logger.info(f"Applying enhanced hub penalty to {len(hub_nodes)} nodes")
        
        for hub_node in hub_nodes:
            hub_degree = node_degrees[hub_node]
            
            if method == 'log_normalization':
                penalty_factor = strength * math.log(hub_degree / threshold + 1)
            elif method == 'sqrt_normalization':
                penalty_factor = strength * math.sqrt(hub_degree / threshold)
            elif method == 'linear_normalization':
                penalty_factor = strength * (hub_degree / threshold)
            else:
                penalty_factor = strength * math.log(hub_degree / threshold + 1)
            
            normalization = 1.0 / (1.0 + penalty_factor)
            
            for u, v, data in G.edges(hub_node, data=True):
                original_weight = data.get('weight', 1.0)
                data['weight'] = original_weight * normalization
                data['enhanced_hub_penalty_applied'] = True
                data['penalty_factor'] = penalty_factor
                data['hub_normalization'] = normalization
            
            if G.has_node(hub_node):
                G.nodes[hub_node]['enhanced_hub_penalty'] = True
                G.nodes[hub_node]['penalty_factor'] = penalty_factor
        
        return G
    
    def _apply_metapath_filtering(self, G: nx.Graph) -> nx.Graph:
        """应用metapath白名单和路径偏好"""
        if G.number_of_edges() == 0:
            return G
        
        whitelist = self.metapath_config['whitelist']
        path_preferences = self.metapath_config['path_preferences']
        max_path_length = self.metapath_config['max_path_length']
        
        if not whitelist and not path_preferences:
            logger.info("No metapath rules configured, skipping filtering")
            return G
        
        edges_to_remove = []
        edges_boosted = 0
        
        for u, v, data in G.edges(data=True):
            relation_type = data.get('relation_type', 'unknown')
            
            # 检查白名单
            if whitelist:
                if relation_type not in whitelist:
                    edges_to_remove.append((u, v))
                    continue
            
            # 应用路径偏好权重
            if path_preferences and relation_type in path_preferences:
                preference_weight = path_preferences[relation_type]
                original_weight = data.get('weight', 1.0)
                data['weight'] = original_weight * preference_weight
                data['metapath_preference_applied'] = True
                data['preference_weight'] = preference_weight
                edges_boosted += 1
        
        # 移除不在白名单中的边
        for u, v in edges_to_remove:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
        
        logger.info(f"Metapath filtering: removed {len(edges_to_remove)} edges, boosted {edges_boosted} edges")
        return G

    def build_graph_with_metrics(self, atomic_notes: List[Dict[str, Any]], embeddings=None):
        """Build the graph and compute quality metrics."""
        G = self.build_graph(atomic_notes, embeddings)
        metrics = {}
        if compute_metrics:
            try:
                metrics = compute_metrics(G)
                logger.info(f"Graph metrics: {metrics}")
            except Exception as exc:  # pragma: no cover - shouldn't break build
                logger.error(f"Failed to compute graph metrics: {exc}")
        return G, metrics


if __name__ == "__main__":
    import argparse
    import json
    import numpy as np
    from utils.file_utils import FileUtils

    parser = argparse.ArgumentParser(description="Build graph and report metrics")
    parser.add_argument("notes", help="Path to atomic notes JSON file")
    parser.add_argument("--embeddings", help="Optional embeddings .npy file")
    args = parser.parse_args()

    notes = FileUtils.read_json(args.notes)
    embeddings = None
    if args.embeddings:
        try:
            embeddings = np.load(args.embeddings)
        except Exception as exc:  # pragma: no cover - optional file
            logger.error(f"Failed to load embeddings: {exc}")

    builder = GraphBuilder()
    graph, metrics = builder.build_graph_with_metrics(notes, embeddings)
    print(json.dumps(metrics, indent=2))
