import re
from typing import List, Dict, Any, Set, Tuple
from loguru import logger
from collections import defaultdict, Counter
from config import config

class ConsistencyChecker:
    """一致性检查器，用于检测原子笔记与知识图谱之间的对齐问题"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def check_consistency(self, atomic_notes: List[Dict[str, Any]], 
                         graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行完整的一致性检查"""
        logger.info("Starting consistency check for atomic notes and graph data")
        
        self.errors = []
        self.warnings = []
        self.stats = {}
        
        # 获取启用的检查项
        check_config = config.get('consistency_check', {})
        
        # 检查1: note_id一致性
        if check_config.get('check_note_id_consistency', True):
            self._check_note_id_consistency(atomic_notes, graph_data)
        
        # 检查2: 实体对齐
        if check_config.get('check_entity_alignment', True):
            self._check_entity_alignment(atomic_notes, graph_data)
        
        # 检查3: 关系链完整性
        if check_config.get('check_relation_integrity', True):
            self._check_relation_chain_integrity(atomic_notes, graph_data)
        
        # 检查4: 源文档绑定稳定性
        if check_config.get('check_source_binding', True):
            self._check_source_binding_stability(atomic_notes)
        
        # 检查5: 图结构完整性
        if check_config.get('check_graph_structure', True):
            self._check_graph_structure_integrity(graph_data)
        
        # 生成统计信息
        self._generate_statistics(atomic_notes, graph_data)
        
        result = {
            'is_consistent': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': self.stats,
            'recommendations': self._generate_recommendations()
        }
        
        logger.info(f"Consistency check completed: {len(self.errors)} errors, {len(self.warnings)} warnings")
        return result
    
    def _check_note_id_consistency(self, atomic_notes: List[Dict[str, Any]], 
                                  graph_data: Dict[str, Any]):
        """检查note_id在原子笔记和图数据中的一致性"""
        note_ids_in_notes = set()
        note_ids_in_graph = set()
        
        # 收集原子笔记中的note_id
        for note in atomic_notes:
            note_id = note.get('note_id')
            if note_id:
                if note_id in note_ids_in_notes:
                    self.errors.append({
                        'type': 'duplicate_note_id',
                        'message': f'Duplicate note_id found: {note_id}',
                        'note_id': note_id
                    })
                note_ids_in_notes.add(note_id)
            else:
                self.errors.append({
                    'type': 'missing_note_id',
                    'message': 'Atomic note missing note_id',
                    'note_content': note.get('content', '')[:100]
                })
        
        # 收集图数据中的note_id
        nodes = graph_data.get('nodes', [])
        for node in nodes:
            node_id = node.get('id')
            if node_id:
                note_ids_in_graph.add(node_id)
        
        # 检查不匹配的note_id
        missing_in_graph = note_ids_in_notes - note_ids_in_graph
        missing_in_notes = note_ids_in_graph - note_ids_in_notes
        
        for note_id in missing_in_graph:
            self.errors.append({
                'type': 'note_missing_in_graph',
                'message': f'Note {note_id} exists in atomic_notes but missing in graph',
                'note_id': note_id
            })
        
        for note_id in missing_in_notes:
            self.errors.append({
                'type': 'graph_node_missing_note',
                'message': f'Graph node {note_id} has no corresponding atomic note',
                'note_id': note_id
            })
    
    def _check_entity_alignment(self, atomic_notes: List[Dict[str, Any]], 
                               graph_data: Dict[str, Any]):
        """检查实体在笔记和图中的对齐情况"""
        note_entities = defaultdict(set)
        graph_entities = defaultdict(set)
        
        # 收集笔记中的实体
        for note in atomic_notes:
            note_id = note.get('note_id')
            if note_id:
                entities = note.get('entities', [])
                if isinstance(entities, list):
                    note_entities[note_id].update(entities)
        
        # 收集图中的实体（从关系元数据中）
        edges = graph_data.get('links', [])
        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            metadata = edge.get('metadata', {})
            
            if source_id and 'entity1' in metadata:
                graph_entities[source_id].add(metadata['entity1'])
            if source_id and 'entity2' in metadata:
                graph_entities[source_id].add(metadata['entity2'])
            if target_id and 'entity1' in metadata:
                graph_entities[target_id].add(metadata['entity1'])
            if target_id and 'entity2' in metadata:
                graph_entities[target_id].add(metadata['entity2'])
        
        # 检查实体对齐
        for note_id in note_entities:
            note_ents = note_entities[note_id]
            graph_ents = graph_entities.get(note_id, set())
            
            # 检查是否有实体在笔记中但不在图关系中
            missing_in_graph = note_ents - graph_ents
            # 从配置中获取阈值，默认80%
            threshold = config.get('consistency_check.entity_alignment_threshold', 0.8)
            if missing_in_graph and len(missing_in_graph) > len(note_ents) * threshold:  # 超过阈值的实体缺失
                self.warnings.append({
                    'type': 'entity_alignment_mismatch',
                    'message': f'Note {note_id} has entities not reflected in graph relations',
                    'note_id': note_id,
                    'missing_entities': list(missing_in_graph)
                })
    
    def _check_relation_chain_integrity(self, atomic_notes: List[Dict[str, Any]], 
                                       graph_data: Dict[str, Any]):
        """检查关系链的完整性，特别是事件链"""
        # 识别可能的事件链关键词
        event_keywords = {
            'succession': ['继任', '接任', '接替', '继承', 'succeed', 'replace'],
            'acquisition': ['收购', '并购', '兼并', 'acquire', 'purchase'],
            'ownership': ['拥有', '持有', '控制', 'own', 'control'],
            'bankruptcy': ['破产', '倒闭', '清算', 'bankruptcy', 'liquidation']
        }
        
        # 查找包含事件关键词的笔记
        event_notes = []
        for note in atomic_notes:
            note_id = note.get('note_id')
            content = note.get('content', '') + ' ' + note.get('original_text', '')
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in content for keyword in keywords):
                    event_notes.append({
                        'note_id': note_id,
                        'event_type': event_type,
                        'content': content
                    })
                    break
        
        # 检查事件笔记是否有对应的图关系
        edges = graph_data.get('links', [])
        edge_map = defaultdict(list)
        
        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            relation_type = edge.get('relation_type', '')
            
            if source_id:
                edge_map[source_id].append({
                    'target': target_id,
                    'type': relation_type,
                    'edge': edge
                })
        
        # 检查事件笔记的关系连接
        for event_note in event_notes:
            note_id = event_note['note_id']
            event_type = event_note['event_type']
            
            related_edges = edge_map.get(note_id, [])
            
            # 检查是否有相应类型的关系
            has_matching_relation = any(
                edge['type'] == event_type for edge in related_edges
            )
            
            if not has_matching_relation and len(related_edges) == 0:
                self.warnings.append({
                    'type': 'isolated_event_note',
                    'message': f'Event note {note_id} ({event_type}) has no graph relations',
                    'note_id': note_id,
                    'event_type': event_type
                })
    
    def _check_source_binding_stability(self, atomic_notes: List[Dict[str, Any]]):
        """检查源文档绑定的稳定性"""
        source_bindings = defaultdict(list)
        
        for note in atomic_notes:
            note_id = note.get('note_id')
            source_info = note.get('source_info', {})
            
            if note_id and source_info:
                file_path = source_info.get('file_path')
                chunk_index = source_info.get('chunk_index')
                
                if file_path and chunk_index is not None:
                    binding_key = f"{file_path}:{chunk_index}"
                    source_bindings[binding_key].append(note_id)
        
        # 检查是否有多个note_id绑定到同一个源位置
        for binding_key, note_ids in source_bindings.items():
            if len(note_ids) > 1:
                self.warnings.append({
                    'type': 'multiple_notes_same_source',
                    'message': f'Multiple notes bound to same source: {binding_key}',
                    'binding_key': binding_key,
                    'note_ids': note_ids
                })
    
    def _check_graph_structure_integrity(self, graph_data: Dict[str, Any]):
        """检查图结构的完整性"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('links', [])
        
        node_ids = set(node.get('id') for node in nodes if node.get('id'))
        
        # 检查边是否引用了不存在的节点
        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            
            if source_id and source_id not in node_ids:
                self.errors.append({
                    'type': 'edge_references_missing_node',
                    'message': f'Edge references non-existent source node: {source_id}',
                    'source_id': source_id,
                    'edge': edge
                })
            
            if target_id and target_id not in node_ids:
                self.errors.append({
                    'type': 'edge_references_missing_node',
                    'message': f'Edge references non-existent target node: {target_id}',
                    'target_id': target_id,
                    'edge': edge
                })
        
        # 检查孤立节点
        connected_nodes = set()
        for edge in edges:
            source_id = edge.get('source')
            target_id = edge.get('target')
            if source_id:
                connected_nodes.add(source_id)
            if target_id:
                connected_nodes.add(target_id)
        
        isolated_nodes = node_ids - connected_nodes
        if len(isolated_nodes) > len(node_ids) * 0.3:  # 超过30%的节点孤立
            self.warnings.append({
                'type': 'high_isolated_nodes_ratio',
                'message': f'High ratio of isolated nodes: {len(isolated_nodes)}/{len(node_ids)}',
                'isolated_count': len(isolated_nodes),
                'total_count': len(node_ids)
            })
    
    def _generate_statistics(self, atomic_notes: List[Dict[str, Any]], 
                           graph_data: Dict[str, Any]):
        """生成统计信息"""
        self.stats = {
            'total_notes': len(atomic_notes),
            'total_nodes': len(graph_data.get('nodes', [])),
            'total_edges': len(graph_data.get('links', [])),
            'notes_with_note_id': sum(1 for note in atomic_notes if note.get('note_id')),
            'notes_with_entities': sum(1 for note in atomic_notes if note.get('entities')),
            'notes_with_source_info': sum(1 for note in atomic_notes if note.get('source_info')),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
        
        # 计算覆盖率
        if self.stats['total_notes'] > 0:
            self.stats['note_id_coverage'] = self.stats['notes_with_note_id'] / self.stats['total_notes']
            self.stats['entity_coverage'] = self.stats['notes_with_entities'] / self.stats['total_notes']
            self.stats['source_binding_coverage'] = self.stats['notes_with_source_info'] / self.stats['total_notes']
    
    def _generate_recommendations(self) -> List[str]:
        """基于检查结果生成修复建议"""
        recommendations = []
        
        if any(error['type'] == 'duplicate_note_id' for error in self.errors):
            recommendations.append("重新生成note_id以确保唯一性")
        
        if any(error['type'] == 'note_missing_in_graph' for error in self.errors):
            recommendations.append("重新构建知识图谱以包含所有原子笔记")
        
        if any(warning['type'] == 'entity_alignment_mismatch' for warning in self.warnings):
            recommendations.append("优化实体抽取和关系识别的一致性")
        
        if any(warning['type'] == 'isolated_event_note' for warning in self.warnings):
            recommendations.append("加强事件链的关系连接，确保事件笔记不孤立")
        
        if any(warning['type'] == 'high_isolated_nodes_ratio' for warning in self.warnings):
            recommendations.append("优化关系抽取策略，减少孤立节点")
        
        if not recommendations:
            recommendations.append("数据一致性良好，无需特殊修复")
        
        return recommendations
    
    def export_report(self, output_path: str):
        """导出一致性检查报告"""
        report = {
            'timestamp': self._get_timestamp(),
            'summary': {
                'is_consistent': len(self.errors) == 0,
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings)
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': self.stats,
            'recommendations': self._generate_recommendations()
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Consistency check report exported to: {output_path}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()