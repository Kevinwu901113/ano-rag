import os
import networkx as nx
from typing import Dict, Any, Optional
from loguru import logger


class GraphMLExporter:
    """专门用于将知识图谱导出为GraphML格式的工具类"""
    
    def __init__(self, graph_index=None):
        """初始化GraphML导出器
        
        Args:
            graph_index: GraphIndex实例，可选
        """
        self.graph_index = graph_index
    
    def export_graph(self, graph: nx.Graph, filepath: str, 
                    centrality_scores: Optional[Dict[str, float]] = None) -> None:
        """将NetworkX图导出为GraphML格式
        
        Args:
            graph: 要导出的NetworkX图
            filepath: 输出文件路径
            centrality_scores: 节点中心性分数字典，可选
        """
        try:
            # 确保文件路径以.graphml结尾
            if not filepath.endswith('.graphml'):
                filepath = filepath + '.graphml'
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 创建一个图的副本用于导出
            export_graph = graph.copy()
            
            # 添加中心性分数（如果提供）
            if centrality_scores:
                for node_id in export_graph.nodes():
                    centrality = centrality_scores.get(node_id, 0.0)
                    export_graph.nodes[node_id]['centrality'] = centrality
            
            # 清理节点属性，确保GraphML兼容性
            self._clean_node_attributes(export_graph)
            
            # 清理边属性，确保GraphML兼容性
            self._clean_edge_attributes(export_graph)
            
            # 保存为GraphML格式
            nx.write_graphml(export_graph, filepath, encoding='utf-8')
            
            # 验证文件完整性
            self._verify_graphml_file(filepath)
            
            logger.info(f"Graph exported to GraphML format: {filepath}")
            logger.info(f"GraphML contains {export_graph.number_of_nodes()} nodes and {export_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to export graph to GraphML format: {e}")
            raise
    
    def export_from_index(self, filepath: str) -> None:
        """从GraphIndex导出GraphML
        
        Args:
            filepath: 输出文件路径
        """
        if not self.graph_index:
            raise ValueError("GraphIndex not provided")
        
        self.export_graph(
            graph=self.graph_index.graph,
            filepath=filepath,
            centrality_scores=self.graph_index.centrality_scores
        )
    
    def _clean_node_attributes(self, graph: nx.Graph) -> None:
        """清理节点属性，确保GraphML兼容性"""
        for node_id, node_data in graph.nodes(data=True):
            for key, value in list(node_data.items()):
                if isinstance(value, (list, dict, tuple)):
                    # 将复杂类型转换为字符串
                    graph.nodes[node_id][key] = str(value)
                elif value is None:
                    # 移除None值
                    del graph.nodes[node_id][key]
                elif isinstance(value, bool):
                    # 确保布尔值被正确处理
                    graph.nodes[node_id][key] = str(value).lower()
    
    def _clean_edge_attributes(self, graph: nx.Graph) -> None:
        """清理边属性，确保GraphML兼容性"""
        for u, v, edge_data in graph.edges(data=True):
            for key, value in list(edge_data.items()):
                if isinstance(value, (list, dict, tuple)):
                    # 将复杂类型转换为字符串
                    graph.edges[u, v][key] = str(value)
                elif value is None:
                    # 移除None值
                    del graph.edges[u, v][key]
                elif isinstance(value, bool):
                    # 确保布尔值被正确处理
                    graph.edges[u, v][key] = str(value).lower()
    
    def _verify_graphml_file(self, filepath: str) -> None:
        """验证GraphML文件的完整性"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有正确的XML声明和GraphML标签
            if not content.startswith('<?xml'):
                raise ValueError("Missing XML declaration")
            
            if '<graphml' not in content:
                raise ValueError("Missing GraphML opening tag")
            
            if not content.rstrip().endswith('</graphml>'):
                logger.warning(f"GraphML file {filepath} appears to be truncated, attempting to fix...")
                # 尝试修复文件
                if not content.rstrip().endswith('</graph>'):
                    content += '\n</graph>'
                if not content.rstrip().endswith('</graphml>'):
                    content += '\n</graphml>'
                
                # 重写文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Fixed truncated GraphML file: {filepath}")
            
            logger.debug(f"GraphML file verification passed: {filepath}")
            
        except Exception as e:
            logger.error(f"GraphML file verification failed for {filepath}: {e}")
            raise
    
    def export_with_metadata(self, graph: nx.Graph, filepath: str, 
                           metadata: Dict[str, Any]) -> None:
        """导出带有额外元数据的GraphML
        
        Args:
            graph: 要导出的NetworkX图
            filepath: 输出文件路径
            metadata: 额外的元数据信息
        """
        try:
            # 确保文件路径以.graphml结尾
            if not filepath.endswith('.graphml'):
                filepath = filepath + '.graphml'
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 创建一个图的副本用于导出
            export_graph = graph.copy()
            
            # 添加图级别的元数据
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    export_graph.graph[key] = value
                else:
                    export_graph.graph[key] = str(value)
            
            # 清理属性
            self._clean_node_attributes(export_graph)
            self._clean_edge_attributes(export_graph)
            
            # 保存为GraphML格式
            nx.write_graphml(export_graph, filepath, encoding='utf-8')
            
            # 验证文件完整性
            self._verify_graphml_file(filepath)
            
            logger.info(f"Graph with metadata exported to GraphML: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export graph with metadata to GraphML: {e}")
            raise