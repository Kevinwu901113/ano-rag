import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """配置加载器，用于加载和管理系统配置"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = config_path
        self._config = None
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key: str, default=None):
        """获取配置项，支持点号分隔的嵌套键"""
        config = self.load_config()
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        config = self.load_config()
        config.update(updates)
        self._config = config
        
    def save_config(self):
        """保存配置到文件"""
        if self._config:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def create_directories(self):
        """创建配置中指定的目录"""
        config = self.load_config()
        storage_paths = config.get('storage', {})
        
        for path_key, path_value in storage_paths.items():
            if path_value and isinstance(path_value, str):
                Path(path_value).mkdir(parents=True, exist_ok=True)
                
# 全局配置实例
config = ConfigLoader()