import os
import json
import jsonlines
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from docx import Document
from loguru import logger

class FileUtils:
    """文件处理工具类，用于处理不同格式的文档"""
    
    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """读取JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """读取JSONL文件"""
        data = []
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
        return data
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """读取Word文档"""
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    
    @staticmethod
    def read_document(file_path: str) -> Union[str, Dict, List]:
        """根据文件扩展名读取文档"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            return FileUtils.read_json(file_path)
        elif ext == '.jsonl':
            return FileUtils.read_jsonl(file_path)
        elif ext == '.docx':
            return FileUtils.read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: str):
        """写入JSON文件"""
        
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                # 转换字典的键和值
                converted_dict = {}
                for key, value in obj.items():
                    # 转换键的类型
                    if isinstance(key, np.integer):
                        converted_key = int(key)
                    elif isinstance(key, np.floating):
                        converted_key = float(key)
                    else:
                        converted_key = key
                    # 转换值的类型
                    converted_dict[converted_key] = convert_numpy_types(value)
                return converted_dict
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 转换数据中的numpy类型
        converted_data = convert_numpy_types(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def write_jsonl(data: List[Dict[str, Any]], file_path: str):
        """写入JSONL文件"""
        with jsonlines.open(file_path, 'w') as writer:
            writer.write_all(data)
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """获取文件的哈希值，用于判断文件是否更改"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def is_file_modified(file_path: str, hash_cache: Dict[str, str]) -> bool:
        """判断文件是否被修改"""
        if not os.path.exists(file_path):
            return False
        
        current_hash = FileUtils.get_file_hash(file_path)
        previous_hash = hash_cache.get(file_path)
        
        if previous_hash is None or current_hash != previous_hash:
            hash_cache[file_path] = current_hash
            return True
        return False
    
    @staticmethod
    def update_hash_cache(hash_cache: Dict[str, str], cache_file: str):
        """更新哈希缓存文件"""
        FileUtils.write_json(hash_cache, cache_file)
    
    @staticmethod
    def load_hash_cache(cache_file: str) -> Dict[str, str]:
        """加载哈希缓存文件"""
        if os.path.exists(cache_file):
            return FileUtils.read_json(cache_file)
        return {}
    
    @staticmethod
    def ensure_dir(directory: str):
        """确保目录存在"""
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def list_files(directory: str, extensions: List[str]) -> List[str]:
        """列出目录中指定扩展名的文件"""
        files = []
        path = Path(directory)
        for ext in extensions:
            files.extend(str(p) for p in path.glob(f"*{ext}"))
        return sorted(files)
