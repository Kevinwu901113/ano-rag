import torch
import numpy as np
from typing import Optional, Union, List
from loguru import logger

# 检查CUDF可用性
CUDF_AVAILABLE = False
try:
    import cudf
    import cuml
    import cugraph
    # 测试基本功能
    test_df = cudf.DataFrame({'test': [1, 2, 3]})
    CUDF_AVAILABLE = True
    logger.info(f"CUDF available - version: {cudf.__version__}")
except ImportError as e:
    logger.info(f"CUDF not available: {e}")
except Exception as e:
    logger.warning(f"CUDF import failed: {e}, falling back to CPU processing")
    try:
        import importlib.metadata as metadata
        cupy_pkgs = [dist.metadata['name'] for dist in metadata.distributions()
                     if dist.metadata['name'].lower().startswith("cupy")]
        if len(cupy_pkgs) > 1:
            logger.warning(
                f"Multiple cupy packages detected: {cupy_pkgs}. "
                "Remove conflicts and reinstall RAPIDS."
            )
    except Exception as meta_e:
        logger.debug(f"Failed to check cupy packages: {meta_e}")

class GPUUtils:
    """GPU工具类，用于管理CUDA和cuDF的使用"""
    
    @staticmethod
    def is_cuda_available() -> bool:
        """检查CUDA是否可用"""
        return torch.cuda.is_available()
    
    @staticmethod
    def is_cudf_available() -> bool:
        """检查cuDF是否可用"""
        return CUDF_AVAILABLE
    
    @staticmethod
    def get_device() -> str:
        """获取推荐的设备"""
        if GPUUtils.is_cuda_available():
            return "cuda"
        return "cpu"
    
    @staticmethod
    def to_cudf(data: Union[np.ndarray, List, dict], use_gpu: bool = True) -> Union['cudf.DataFrame', 'pd.DataFrame']:
        """将数据转换为cuDF DataFrame（如果可用）"""
        if use_gpu and CUDF_AVAILABLE:
            try:
                if isinstance(data, dict):
                    return cudf.DataFrame(data)
                elif isinstance(data, (list, np.ndarray)):
                    return cudf.DataFrame({'data': data})
                else:
                    return cudf.DataFrame(data)
            except Exception as e:
                logger.warning(f"Failed to create cuDF DataFrame: {e}, falling back to pandas")
        
        # Fallback to pandas
        import pandas as pd
        if isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, (list, np.ndarray)):
            return pd.DataFrame({'data': data})
        else:
            return pd.DataFrame(data)
    
    @staticmethod
    def to_numpy(df: Union['cudf.DataFrame', 'pd.DataFrame']) -> np.ndarray:
        """将DataFrame转换为numpy数组"""
        if CUDF_AVAILABLE and hasattr(df, 'to_pandas'):
            # cuDF DataFrame
            return df.to_pandas().values
        else:
            # pandas DataFrame
            return df.values
    
    @staticmethod
    def batch_process_gpu(
        data: Union[List, "cudf.DataFrame"],
        batch_size: int,
        process_func,
        use_gpu: bool = True,
    ):
        """批处理函数，支持列表或cuDF DataFrame"""

        results = []

        is_cudf_df = CUDF_AVAILABLE and hasattr(data, "iloc")
        data_len = len(data)

        for i in range(0, data_len, batch_size):
            batch = data.iloc[i : i + batch_size] if is_cudf_df else data[i : i + batch_size]

            if use_gpu and CUDF_AVAILABLE:
                try:
                    result = process_func(batch)
                except Exception as e:
                    logger.warning(f"GPU processing failed: {e}, falling back to CPU")
                    cpu_batch = batch.to_pandas() if is_cudf_df and hasattr(batch, "to_pandas") else batch
                    result = process_func(cpu_batch)
            else:
                cpu_batch = batch.to_pandas() if is_cudf_df and hasattr(batch, "to_pandas") else batch
                result = process_func(cpu_batch)

            results.extend(result)

        return results
    
    @staticmethod
    def optimize_clustering(embeddings: np.ndarray, use_gpu: bool = True, algorithm: str = "hdbscan", **kwargs):
        """优化的聚类算法，优先使用GPU"""
        if use_gpu and CUDF_AVAILABLE:
            try:
                # 使用cuML进行GPU聚类
                if algorithm.lower() == "hdbscan":
                    from cuml.cluster import HDBSCAN
                    clusterer = HDBSCAN(
                        min_cluster_size=kwargs.get('min_cluster_size', 5),
                        min_samples=kwargs.get('min_samples', 3),
                        metric=kwargs.get('metric', 'euclidean')
                    )
                elif algorithm.lower() == "kmeans":
                    from cuml.cluster import KMeans
                    clusterer = KMeans(
                        n_clusters=kwargs.get('n_clusters', 8),
                        random_state=42
                    )
                elif algorithm.lower() == "dbscan":
                    from cuml.cluster import DBSCAN
                    clusterer = DBSCAN(
                        eps=kwargs.get('eps', 0.5),
                        min_samples=kwargs.get('min_samples', 5)
                    )
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
                
                # 转换为cuDF
                embeddings_cudf = cudf.DataFrame(embeddings)
                labels = clusterer.fit_predict(embeddings_cudf)
                return labels.to_pandas().values if hasattr(labels, 'to_pandas') else labels
                
            except Exception as e:
                logger.warning(f"GPU clustering failed: {e}, falling back to CPU")
        
        # CPU回退
        from sklearn.cluster import HDBSCAN, KMeans, DBSCAN
        
        if algorithm.lower() == "hdbscan":
            clusterer = HDBSCAN(
                min_cluster_size=kwargs.get('min_cluster_size', 5),
                min_samples=kwargs.get('min_samples', 3),
                metric=kwargs.get('metric', 'euclidean')
            )
        elif algorithm.lower() == "kmeans":
            clusterer = KMeans(
                n_clusters=kwargs.get('n_clusters', 8),
                random_state=42
            )
        elif algorithm.lower() == "dbscan":
            clusterer = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
        
        return clusterer.fit_predict(embeddings)
    
    @staticmethod
    def get_memory_info() -> dict:
        """获取GPU内存信息"""
        if GPUUtils.is_cuda_available():
            return {
                'total': torch.cuda.get_device_properties(0).total_memory,
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved()
            }
        return {'total': 0, 'allocated': 0, 'cached': 0}
