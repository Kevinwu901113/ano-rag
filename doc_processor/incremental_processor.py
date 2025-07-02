import os
import json
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
from loguru import logger
from utils import FileUtils
from config import config

class IncrementalProcessor:
    """增量处理器，用于处理文档的增量更新"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or config.get('storage.cache_path', './data/cache')
        self.hash_cache_file = os.path.join(self.cache_dir, 'file_hashes.json')
        self.processed_files_cache = os.path.join(self.cache_dir, 'processed_files.json')
        self.dependency_cache = os.path.join(self.cache_dir, 'dependencies.json')
        
        # 确保缓存目录存在
        FileUtils.ensure_dir(self.cache_dir)
        
        # 加载缓存
        self.file_hashes = self._load_hash_cache()
        self.processed_files = self._load_processed_files_cache()
        self.dependencies = self._load_dependency_cache()
    
    def check_files_for_changes(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """检查文件是否有变化"""
        logger.info(f"Checking {len(file_paths)} files for changes")
        
        result = {
            'new_files': [],
            'modified_files': [],
            'unchanged_files': [],
            'deleted_files': []
        }
        
        # 检查现有文件
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            if self._is_new_file(file_path):
                result['new_files'].append(file_path)
            elif self._is_file_modified(file_path):
                result['modified_files'].append(file_path)
            else:
                result['unchanged_files'].append(file_path)
        
        # 检查已删除的文件
        result['deleted_files'] = self._find_deleted_files(file_paths)
        
        logger.info(f"Change detection: {len(result['new_files'])} new, "
                   f"{len(result['modified_files'])} modified, "
                   f"{len(result['unchanged_files'])} unchanged, "
                   f"{len(result['deleted_files'])} deleted")
        
        return result
    
    def _is_new_file(self, file_path: str) -> bool:
        """检查是否为新文件"""
        return file_path not in self.file_hashes
    
    def _is_file_modified(self, file_path: str) -> bool:
        """检查文件是否被修改"""
        if file_path not in self.file_hashes:
            return True
        
        current_hash = FileUtils.get_file_hash(file_path)
        cached_hash = self.file_hashes.get(file_path)
        
        return current_hash != cached_hash
    
    def _find_deleted_files(self, current_files: List[str]) -> List[str]:
        """查找已删除的文件"""
        current_files_set = set(current_files)
        cached_files_set = set(self.file_hashes.keys())
        
        deleted_files = []
        for cached_file in cached_files_set:
            if cached_file not in current_files_set and not os.path.exists(cached_file):
                deleted_files.append(cached_file)
        
        return deleted_files
    
    def update_file_cache(self, file_path: str, processing_result: Dict[str, Any] = None):
        """更新文件缓存"""
        # 更新文件哈希
        if os.path.exists(file_path):
            self.file_hashes[file_path] = FileUtils.get_file_hash(file_path)
        
        # 更新处理结果缓存
        if processing_result:
            self.processed_files[file_path] = {
                'last_processed': self._get_timestamp(),
                'result': processing_result
            }
        
        # 保存缓存
        self._save_caches()
    
    def get_cached_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取缓存的处理结果"""
        if file_path in self.processed_files:
            return self.processed_files[file_path].get('result')
        return None
    
    def remove_from_cache(self, file_path: str):
        """从缓存中移除文件"""
        self.file_hashes.pop(file_path, None)
        self.processed_files.pop(file_path, None)
        
        # 移除相关依赖
        self._remove_dependencies(file_path)
        
        self._save_caches()
    
    def batch_update_cache(self, file_results: Dict[str, Dict[str, Any]]):
        """批量更新缓存"""
        for file_path, result in file_results.items():
            self.update_file_cache(file_path, result)
    
    def get_processing_plan(self, file_paths: List[str]) -> Dict[str, Any]:
        """获取处理计划"""
        changes = self.check_files_for_changes(file_paths)
        
        # 需要处理的文件
        files_to_process = changes['new_files'] + changes['modified_files']
        
        # 需要重新处理的依赖文件
        dependent_files = self._get_dependent_files(files_to_process)
        
        # 需要清理的文件
        files_to_clean = changes['deleted_files']
        
        plan = {
            'files_to_process': files_to_process,
            'dependent_files': dependent_files,
            'files_to_clean': files_to_clean,
            'unchanged_files': changes['unchanged_files'],
            'total_files_to_process': len(files_to_process) + len(dependent_files),
            'can_skip_processing': len(files_to_process) == 0 and len(dependent_files) == 0
        }
        
        return plan
    
    def _get_dependent_files(self, changed_files: List[str]) -> List[str]:
        """获取依赖于已更改文件的其他文件"""
        dependent_files = set()
        
        for changed_file in changed_files:
            # 查找依赖于此文件的其他文件
            for file_path, deps in self.dependencies.items():
                if changed_file in deps.get('depends_on', []):
                    dependent_files.add(file_path)
        
        return list(dependent_files)
    
    def add_dependency(self, file_path: str, depends_on: List[str]):
        """添加文件依赖关系"""
        if file_path not in self.dependencies:
            self.dependencies[file_path] = {
                'depends_on': [],
                'dependents': []
            }
        
        self.dependencies[file_path]['depends_on'] = depends_on
        
        # 更新反向依赖
        for dep_file in depends_on:
            if dep_file not in self.dependencies:
                self.dependencies[dep_file] = {
                    'depends_on': [],
                    'dependents': []
                }
            
            if file_path not in self.dependencies[dep_file]['dependents']:
                self.dependencies[dep_file]['dependents'].append(file_path)
    
    def _remove_dependencies(self, file_path: str):
        """移除文件的依赖关系"""
        if file_path in self.dependencies:
            # 移除此文件对其他文件的依赖
            depends_on = self.dependencies[file_path].get('depends_on', [])
            for dep_file in depends_on:
                if dep_file in self.dependencies:
                    dependents = self.dependencies[dep_file].get('dependents', [])
                    if file_path in dependents:
                        dependents.remove(file_path)
            
            # 移除此文件的记录
            del self.dependencies[file_path]
    
    def optimize_processing_order(self, files_to_process: List[str]) -> List[str]:
        """优化处理顺序，考虑依赖关系"""
        # 拓扑排序，确保依赖文件先处理
        processed = set()
        result = []
        
        def visit(file_path):
            if file_path in processed or file_path not in files_to_process:
                return
            
            # 先处理依赖的文件
            deps = self.dependencies.get(file_path, {}).get('depends_on', [])
            for dep in deps:
                if dep in files_to_process:
                    visit(dep)
            
            processed.add(file_path)
            result.append(file_path)
        
        for file_path in files_to_process:
            visit(file_path)
        
        return result
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_files = len(self.file_hashes)
        processed_files = len(self.processed_files)
        
        # 计算缓存大小
        cache_size = 0
        for cache_file in [self.hash_cache_file, self.processed_files_cache, self.dependency_cache]:
            if os.path.exists(cache_file):
                cache_size += os.path.getsize(cache_file)
        
        return {
            'total_cached_files': total_files,
            'processed_files': processed_files,
            'cache_size_bytes': cache_size,
            'cache_size_mb': cache_size / (1024 * 1024),
            'dependency_count': len(self.dependencies)
        }
    
    def clear_cache(self, file_patterns: List[str] = None):
        """清理缓存"""
        if file_patterns is None:
            # 清理所有缓存
            self.file_hashes.clear()
            self.processed_files.clear()
            self.dependencies.clear()
            logger.info("Cleared all cache")
        else:
            # 清理匹配模式的文件
            import fnmatch
            
            files_to_remove = []
            for file_path in self.file_hashes.keys():
                for pattern in file_patterns:
                    if fnmatch.fnmatch(file_path, pattern):
                        files_to_remove.append(file_path)
                        break
            
            for file_path in files_to_remove:
                self.remove_from_cache(file_path)
            
            logger.info(f"Cleared cache for {len(files_to_remove)} files")
        
        self._save_caches()
    
    def _load_hash_cache(self) -> Dict[str, str]:
        """加载文件哈希缓存"""
        if os.path.exists(self.hash_cache_file):
            try:
                return FileUtils.read_json(self.hash_cache_file)
            except Exception as e:
                logger.warning(f"Failed to load hash cache: {e}")
        return {}
    
    def _load_processed_files_cache(self) -> Dict[str, Dict[str, Any]]:
        """加载处理文件缓存"""
        if os.path.exists(self.processed_files_cache):
            try:
                return FileUtils.read_json(self.processed_files_cache)
            except Exception as e:
                logger.warning(f"Failed to load processed files cache: {e}")
        return {}
    
    def _load_dependency_cache(self) -> Dict[str, Dict[str, List[str]]]:
        """加载依赖关系缓存"""
        if os.path.exists(self.dependency_cache):
            try:
                return FileUtils.read_json(self.dependency_cache)
            except Exception as e:
                logger.warning(f"Failed to load dependency cache: {e}")
        return {}
    
    def _save_caches(self):
        """保存所有缓存"""
        try:
            FileUtils.write_json(self.file_hashes, self.hash_cache_file)
            FileUtils.write_json(self.processed_files, self.processed_files_cache)
            FileUtils.write_json(self.dependencies, self.dependency_cache)
        except Exception as e:
            logger.error(f"Failed to save caches: {e}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """验证缓存完整性"""
        issues = {
            'missing_files': [],
            'hash_mismatches': [],
            'broken_dependencies': [],
            'orphaned_cache_entries': []
        }
        
        # 检查文件是否存在
        for file_path in self.file_hashes.keys():
            if not os.path.exists(file_path):
                issues['missing_files'].append(file_path)
            else:
                # 检查哈希是否匹配
                current_hash = FileUtils.get_file_hash(file_path)
                cached_hash = self.file_hashes[file_path]
                if current_hash != cached_hash:
                    issues['hash_mismatches'].append(file_path)
        
        # 检查依赖关系
        for file_path, deps in self.dependencies.items():
            for dep_file in deps.get('depends_on', []):
                if not os.path.exists(dep_file):
                    issues['broken_dependencies'].append({
                        'file': file_path,
                        'missing_dependency': dep_file
                    })
        
        # 检查孤立的缓存条目
        for file_path in self.processed_files.keys():
            if file_path not in self.file_hashes:
                issues['orphaned_cache_entries'].append(file_path)
        
        return issues
    
    def repair_cache(self, validation_result: Dict[str, Any] = None):
        """修复缓存问题"""
        if validation_result is None:
            validation_result = self.validate_cache_integrity()
        
        # 移除缺失的文件
        for file_path in validation_result['missing_files']:
            self.remove_from_cache(file_path)
        
        # 更新哈希不匹配的文件
        for file_path in validation_result['hash_mismatches']:
            if os.path.exists(file_path):
                self.file_hashes[file_path] = FileUtils.get_file_hash(file_path)
        
        # 修复损坏的依赖关系
        for issue in validation_result['broken_dependencies']:
            file_path = issue['file']
            missing_dep = issue['missing_dependency']
            if file_path in self.dependencies:
                deps = self.dependencies[file_path]['depends_on']
                if missing_dep in deps:
                    deps.remove(missing_dep)
        
        # 移除孤立的缓存条目
        for file_path in validation_result['orphaned_cache_entries']:
            self.processed_files.pop(file_path, None)
        
        self._save_caches()
        logger.info("Cache repair completed")