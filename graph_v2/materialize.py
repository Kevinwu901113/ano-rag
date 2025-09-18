"""
Graph materialization utilities
"""

import os
import json
import time
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any


def write_snapshot(edges: List[Dict[str, Any]], notes: List[Dict[str, Any]], 
                  out_dir: str, degree_cap: int = 20) -> str:
    """
    构建无向图：节点=notes.id；边=(src,dst,weight,evidence)
    对每节点做度裁剪（按weight降序保留<=degree_cap）
    输出到 out_dir/<timestamp>/graph.graphml 和 edges.jsonl
    返回快照目录路径
    
    Args:
        edges: 边列表，每条边包含 src, dst, weight, evidence
        notes: 笔记列表，每个笔记包含 id
        out_dir: 输出目录
        degree_cap: 度数上限
    
    Returns:
        快照目录路径
    """
    # 创建时间戳目录
    timestamp = str(int(time.time()))
    snapshot_dir = Path(out_dir) / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 networkx 构建无向图
    G = nx.Graph()
    
    # 添加节点
    note_dict = {note["id"]: note for note in notes}
    for note in notes:
        node_id = note["id"]
        title = note.get("title", f"Node_{node_id}")
        G.add_node(node_id, title=title)
    
    # 添加边到图中，用于度裁剪
    edge_data = {}  # 存储边的详细信息
    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        weight = edge.get("weight", 1.0)
        evidence = edge.get("evidence", {})
        
        # 添加边到图中
        if G.has_edge(src, dst):
            # 如果边已存在，保留权重更高的
            if weight > G[src][dst].get('weight', 0):
                G[src][dst]['weight'] = weight
                edge_data[(src, dst)] = {"weight": weight, "evidence": evidence}
        else:
            G.add_edge(src, dst, weight=weight)
            edge_data[(src, dst)] = {"weight": weight, "evidence": evidence}
    
    # 对每个节点进行度裁剪
    final_edges = []
    for node in G.nodes():
        # 获取该节点的所有邻居及边权重
        neighbors = []
        for neighbor in G.neighbors(node):
            edge_key = (min(node, neighbor), max(node, neighbor))
            if edge_key in edge_data:
                weight = edge_data[edge_key]["weight"]
                evidence = edge_data[edge_key]["evidence"]
                neighbors.append({
                    "neighbor": neighbor,
                    "weight": weight,
                    "evidence": evidence
                })
        
        # 按权重降序排序，保留前 degree_cap 个
        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        neighbors = neighbors[:degree_cap]
        
        # 添加到最终边列表（避免重复）
        for neighbor_info in neighbors:
            neighbor = neighbor_info["neighbor"]
            weight = neighbor_info["weight"]
            evidence = neighbor_info["evidence"]
            
            # 确保边的方向一致（小节点ID在前）
            src, dst = (node, neighbor) if node < neighbor else (neighbor, node)
            edge_dict = {
                "src": src,
                "dst": dst,
                "weight": weight,
                "evidence": evidence
            }
            
            # 避免重复添加
            if edge_dict not in final_edges:
                final_edges.append(edge_dict)
    
    # 写入 edges.jsonl 文件
    edges_file = snapshot_dir / "edges.jsonl"
    with open(edges_file, 'w', encoding='utf-8') as f:
        for edge in final_edges:
            f.write(json.dumps(edge, ensure_ascii=False) + '\n')
    
    # 写入 GraphML 文件
    graphml_file = snapshot_dir / "graph.graphml"
    with open(graphml_file, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        f.write('  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>\n')
        f.write('  <key id="evidence" for="edge" attr.name="evidence" attr.type="string"/>\n')
        f.write('  <key id="title" for="node" attr.name="title" attr.type="string"/>\n')
        f.write('  <graph id="G" edgedefault="undirected">\n')
        
        # 写入节点
        node_ids = set()
        for edge in final_edges:
            node_ids.add(edge["src"])
            node_ids.add(edge["dst"])
        
        for node_id in sorted(node_ids):
            note = note_dict.get(node_id, {})
            title = note.get("title", f"Node_{node_id}")
            # 转义XML特殊字符
            title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            f.write(f'    <node id="{node_id}">\n')
            f.write(f'      <data key="title">{title}</data>\n')
            f.write(f'    </node>\n')
        
        # 写入边
        for i, edge in enumerate(final_edges):
            src = edge["src"]
            dst = edge["dst"]
            weight = edge["weight"]
            evidence = edge["evidence"]
            
            # 将 evidence 转换为 JSON 字符串
            evidence_str = json.dumps(evidence, ensure_ascii=False)
            # 转义XML特殊字符
            evidence_str = evidence_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            
            f.write(f'    <edge id="e{i}" source="{src}" target="{dst}">\n')
            f.write(f'      <data key="weight">{weight}</data>\n')
            f.write(f'      <data key="evidence">{evidence_str}</data>\n')
            f.write(f'    </edge>\n')
        
        f.write('  </graph>\n')
        f.write('</graphml>\n')
    
    return str(snapshot_dir)