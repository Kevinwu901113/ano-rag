import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from sklearn.cluster import HDBSCAN, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from utils import GPUUtils, BatchProcessor
from config import config

class TopicClustering:
    """主题聚类模块，用于对原子笔记进行聚类操作"""
    
    def __init__(self):
        self.algorithm = config.get('clustering.algorithm', 'hdbscan')
        self.min_cluster_size = config.get('clustering.min_cluster_size', 5)
        self.min_samples = config.get('clustering.min_samples', 3)
        self.metric = config.get('clustering.metric', 'euclidean')
        self.use_gpu = config.get('clustering.use_gpu', True)
        self.batch_processor = BatchProcessor(use_gpu=self.use_gpu)
        
    def cluster_notes(self, atomic_notes: List[Dict[str, Any]], embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """对原子笔记进行聚类"""
        logger.info(f"Clustering {len(atomic_notes)} atomic notes using {self.algorithm}")
        
        if len(atomic_notes) < self.min_cluster_size:
            logger.warning("Too few notes for clustering, assigning all to cluster 0")
            return self._assign_single_cluster(atomic_notes)
        
        # 执行聚类
        cluster_labels = self._perform_clustering(embeddings)
        
        # 分析聚类结果
        cluster_info = self._analyze_clusters(cluster_labels, embeddings)
        
        # 为笔记分配聚类标签和主题信息
        clustered_notes = self._assign_cluster_labels(atomic_notes, cluster_labels, cluster_info)
        
        # 生成主题池
        topic_pools = self._create_topic_pools(clustered_notes, cluster_info)
        
        logger.info(f"Created {len(topic_pools)} topic pools")
        
        return {
            'clustered_notes': clustered_notes,
            'topic_pools': topic_pools,
            'cluster_info': cluster_info
        }
    
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """执行聚类算法"""
        try:
            # 使用GPU优化的聚类（如果可用）
            if self.use_gpu and GPUUtils.is_cudf_available():
                logger.info("Using GPU-accelerated clustering")
                return GPUUtils.optimize_clustering(
                    embeddings,
                    use_gpu=True,
                    algorithm=self.algorithm,
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric
                )
            else:
                logger.info("Using CPU clustering")
                return self._cpu_clustering(embeddings)
                
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # 回退到简单的K-means聚类
            return self._fallback_clustering(embeddings)
    
    def _cpu_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """CPU聚类实现"""
        if self.algorithm.lower() == 'hdbscan':
            clusterer = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_epsilon=0.1
            )
        elif self.algorithm.lower() == 'kmeans':
            # 自动确定K值
            optimal_k = self._find_optimal_k(embeddings)
            clusterer = KMeans(
                n_clusters=optimal_k,
                random_state=42,
                n_init=10
            )
        elif self.algorithm.lower() == 'dbscan':
            # 自动确定eps值
            optimal_eps = self._find_optimal_eps(embeddings)
            clusterer = DBSCAN(
                eps=optimal_eps,
                min_samples=self.min_samples,
                metric=self.metric
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
        
        return clusterer.fit_predict(embeddings)
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """备用聚类方法"""
        logger.warning("Using fallback K-means clustering")
        
        # 简单的K-means聚类
        n_clusters = min(8, max(2, len(embeddings) // 10))
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        return clusterer.fit_predict(embeddings)
    
    def _find_optimal_k(self, embeddings: np.ndarray, max_k: int = None) -> int:
        """使用肘部法则找到最优的K值"""
        if max_k is None:
            max_k = min(10, len(embeddings) // 5)
        
        if max_k < 2:
            return 2
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(embeddings)
                inertias.append(kmeans.inertia_)
            except Exception:
                break
        
        if len(inertias) < 2:
            return 2
        
        # 计算肘部点
        optimal_k = self._find_elbow_point(list(k_range)[:len(inertias)], inertias)
        return optimal_k
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """找到肘部点"""
        if len(k_values) < 3:
            return k_values[0] if k_values else 2
        
        # 计算二阶导数来找肘部点
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # 找到二阶导数最大的点
        elbow_idx = np.argmax(second_diffs) + 1
        return k_values[elbow_idx]
    
    def _find_optimal_eps(self, embeddings: np.ndarray) -> float:
        """为DBSCAN找到最优的eps值"""
        from sklearn.neighbors import NearestNeighbors
        
        # 计算k-distance图
        k = self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # 取第k个最近邻的距离
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # 找到距离急剧增加的点
        diffs = np.diff(k_distances)
        knee_idx = np.argmax(diffs)
        
        optimal_eps = k_distances[knee_idx]
        
        # 确保eps在合理范围内
        return max(0.1, min(2.0, optimal_eps))
    
    def _analyze_clusters(self, cluster_labels: np.ndarray, embeddings: np.ndarray) -> Dict[str, Any]:
        """分析聚类结果"""
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # 排除噪声点(-1)
        n_noise = np.sum(cluster_labels == -1)
        
        cluster_info = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': {},
            'cluster_centers': {},
            'cluster_quality': {}
        }
        
        # 计算每个聚类的信息
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
                
            cluster_mask = cluster_labels == label
            cluster_embeddings = embeddings[cluster_mask]
            
            cluster_info['cluster_sizes'][label] = int(np.sum(cluster_mask))
            cluster_info['cluster_centers'][label] = np.mean(cluster_embeddings, axis=0).tolist()
            
            # 计算聚类内的紧密度
            if len(cluster_embeddings) > 1:
                from scipy.spatial.distance import pdist
                distances = pdist(cluster_embeddings, metric='cosine')
                cluster_info['cluster_quality'][label] = {
                    'cohesion': float(np.mean(distances)),
                    'std': float(np.std(distances))
                }
        
        # 计算整体聚类质量
        if n_clusters > 1 and len(embeddings) > n_clusters:
            try:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                cluster_info['silhouette_score'] = float(silhouette_avg)
            except Exception:
                cluster_info['silhouette_score'] = 0.0
        else:
            cluster_info['silhouette_score'] = 0.0
        
        return cluster_info
    
    def _assign_cluster_labels(self, atomic_notes: List[Dict[str, Any]], 
                              cluster_labels: np.ndarray, 
                              cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为原子笔记分配聚类标签"""
        clustered_notes = []
        
        for i, note in enumerate(atomic_notes):
            note_copy = note.copy()
            cluster_label = int(cluster_labels[i])
            
            note_copy['cluster_id'] = cluster_label
            note_copy['is_noise'] = cluster_label == -1
            
            if cluster_label >= 0:
                note_copy['cluster_size'] = cluster_info['cluster_sizes'].get(cluster_label, 0)
                note_copy['cluster_quality'] = cluster_info['cluster_quality'].get(cluster_label, {})
            
            clustered_notes.append(note_copy)
        
        return clustered_notes
    
    def _create_topic_pools(self, clustered_notes: List[Dict[str, Any]], 
                           cluster_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建主题池"""
        topic_pools = []
        
        # 按聚类分组
        cluster_groups = {}
        for note in clustered_notes:
            cluster_id = note['cluster_id']
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(note)
        
        # 为每个聚类创建主题池
        for cluster_id, notes in cluster_groups.items():
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            topic_pool = self._create_single_topic_pool(cluster_id, notes, cluster_info)
            topic_pools.append(topic_pool)
        
        return topic_pools
    
    def _create_single_topic_pool(self, cluster_id: int, 
                                 notes: List[Dict[str, Any]], 
                                 cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """创建单个主题池"""
        # 提取所有关键词和实体
        all_keywords = []
        all_entities = []
        all_concepts = []
        
        for note in notes:
            all_keywords.extend(note.get('keywords', []))
            all_entities.extend(note.get('entities', []))
            all_concepts.extend(note.get('concepts', []))
        
        # 统计频率并选择最重要的
        top_keywords = self._get_top_items(all_keywords, top_k=10)
        top_entities = self._get_top_items(all_entities, top_k=8)
        top_concepts = self._get_top_items(all_concepts, top_k=5)
        
        # 生成主题描述
        topic_description = self._generate_topic_description(top_keywords, top_entities, top_concepts)
        
        topic_pool = {
            'topic_id': f"topic_{cluster_id:03d}",
            'cluster_id': cluster_id,
            'note_count': len(notes),
            'note_ids': [note.get('note_id', '') for note in notes],
            'topic_keywords': top_keywords,
            'topic_entities': top_entities,
            'topic_concepts': top_concepts,
            'topic_description': topic_description,
            'cluster_center': cluster_info['cluster_centers'].get(cluster_id, []),
            'cluster_quality': cluster_info['cluster_quality'].get(cluster_id, {}),
            'created_at': self._get_timestamp()
        }
        
        return topic_pool
    
    def _get_top_items(self, items: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """获取频率最高的项目"""
        from collections import Counter
        
        # 清理和统计
        cleaned_items = [item.strip() for item in items if item and item.strip()]
        counter = Counter(cleaned_items)
        
        # 获取top_k项目
        top_items = []
        for item, count in counter.most_common(top_k):
            top_items.append({
                'term': item,
                'frequency': count,
                'importance': count / len(cleaned_items) if cleaned_items else 0
            })
        
        return top_items
    
    def _generate_topic_description(self, keywords: List[Dict], 
                                   entities: List[Dict], 
                                   concepts: List[Dict]) -> str:
        """生成主题描述"""
        description_parts = []
        
        if keywords:
            top_keywords = [item['term'] for item in keywords[:3]]
            description_parts.append(f"关键词: {', '.join(top_keywords)}")
        
        if entities:
            top_entities = [item['term'] for item in entities[:3]]
            description_parts.append(f"实体: {', '.join(top_entities)}")
        
        if concepts:
            top_concepts = [item['term'] for item in concepts[:2]]
            description_parts.append(f"概念: {', '.join(top_concepts)}")
        
        return '; '.join(description_parts) if description_parts else "未分类主题"
    
    def _assign_single_cluster(self, atomic_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将所有笔记分配到单个聚类"""
        clustered_notes = []
        for note in atomic_notes:
            note_copy = note.copy()
            note_copy['cluster_id'] = 0
            note_copy['is_noise'] = False
            note_copy['cluster_size'] = len(atomic_notes)
            clustered_notes.append(note_copy)
        
        # 创建单个主题池
        topic_pool = self._create_single_topic_pool(0, clustered_notes, {
            'cluster_centers': {0: []},
            'cluster_quality': {0: {'cohesion': 0.5, 'std': 0.1}}
        })
        
        return {
            'clustered_notes': clustered_notes,
            'topic_pools': [topic_pool],
            'cluster_info': {
                'n_clusters': 1,
                'n_noise': 0,
                'silhouette_score': 0.0
            }
        }
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def optimize_clustering_parameters(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """优化聚类参数"""
        logger.info("Optimizing clustering parameters")
        
        best_params = {
            'algorithm': self.algorithm,
            'score': -1,
            'params': {}
        }
        
        # 测试不同的参数组合
        if self.algorithm.lower() == 'hdbscan':
            min_cluster_sizes = [3, 5, 8, 10]
            min_samples_list = [2, 3, 5]
            
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_list:
                    if min_samples <= min_cluster_size:
                        try:
                            clusterer = HDBSCAN(
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=self.metric
                            )
                            labels = clusterer.fit_predict(embeddings)
                            
                            if len(np.unique(labels[labels >= 0])) > 1:
                                score = silhouette_score(embeddings, labels)
                                
                                if score > best_params['score']:
                                    best_params['score'] = score
                                    best_params['params'] = {
                                        'min_cluster_size': min_cluster_size,
                                        'min_samples': min_samples
                                    }
                        except Exception:
                            continue
        
        return best_params