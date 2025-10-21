import os
import json
import fcntl
import jsonlines
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from docx import Document
from loguru import logger

class FileUtils:
    """文件处理工具类，用于处理不同格式的文档"""

    # ---------------------------
    # 通用：类型转换
    # ---------------------------
    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        """递归转换 numpy 类型为 Python 原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            converted_dict = {}
            for key, value in obj.items():
                if isinstance(key, np.integer):
                    converted_key = int(key)
                elif isinstance(key, np.floating):
                    converted_key = float(key)
                else:
                    converted_key = key
                converted_dict[converted_key] = FileUtils._convert_numpy_types(value)
            return converted_dict
        if isinstance(obj, list):
            return [FileUtils._convert_numpy_types(item) for item in obj]
        return obj

    # ---------------------------
    # 读取
    # ---------------------------
    @staticmethod
    def read_json(file_path: str) -> Any:
        """读取 JSON 文件（可能返回 dict 或 list）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """读取 JSONL 文件"""
        data: List[Dict[str, Any]] = []
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                data.append(item)
        return data

    @staticmethod
    def read_docx(file_path: str) -> str:
        """读取 Word 文档为纯文本"""
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)

    @staticmethod
    def read_document(file_path: str) -> Union[str, Dict, List]:
        """根据扩展名读取文档"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            return FileUtils.read_json(file_path)
        elif ext == '.jsonl':
            return FileUtils.read_jsonl(file_path)
        elif ext == '.docx':
            return FileUtils.read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    # ---------------------------
    # 写入
    # ---------------------------
    @staticmethod
    def write_json(data: Any, file_path: str):
        """写入 JSON 文件（自动转换 numpy 类型）"""
        converted = FileUtils._convert_numpy_types(data)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_jsonl(data: List[Dict[str, Any]], file_path: str):
        """写入 JSONL 文件"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(file_path, 'w') as writer:
            writer.write_all([FileUtils._convert_numpy_types(x) for x in data])

    @staticmethod
    def write_file(file_path: str, content: Optional[str]):
        """写入纯文本文件"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content or "")

    @staticmethod
    def append_jsonl_atomic(file_path: str, record: Dict[str, Any]):
        """
        原子性地向 JSONL 追加单行记录（POSIX 下使用 fcntl.flock）
        NOTE: Windows 环境需要用 portalocker 或其他方式替代。
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(FileUtils._convert_numpy_types(record), ensure_ascii=False)
        with open(path, 'a', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(serialized)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    @staticmethod
    def write_manifest(file_path: str, manifest: Dict[str, Any]):
        """写入运行清单文件"""
        FileUtils.write_json(manifest, file_path)

    # ---------------------------
    # 文件信息 & 缓存
    # ---------------------------
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """获取文件 MD5（变更检测用）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def sha1sum(file_path: str) -> str:
        """获取文件 SHA1（签名/指纹用）"""
        hash_sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha1.update(chunk)
        return hash_sha1.hexdigest()

    @staticmethod
    def is_file_modified(file_path: str, hash_cache: Dict[str, str]) -> bool:
        """
        判断文件是否修改：
        - 缓存无记录 → 视为已修改（返回 True）
        - hash 不同 → True；相同 → False
        - 文件不存在 → False（按需改成 True 也可以）
        """
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
    def get_file_size_bytes(file_path: str) -> int:
        """获取文件大小（字节）"""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found when calculating size: {file_path}")
            return 0
        return path.stat().st_size

    @staticmethod
    def count_file_lines(file_path: str) -> int:
        """统计文件行数（仅在需要时使用）"""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found when counting lines: {file_path}")
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    # ---------------------------
    # 目录操作
    # ---------------------------
    @staticmethod
    def ensure_dir(directory: str):
        """确保目录存在"""
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def list_files(directory: str, extensions: List[str]) -> List[str]:
        """列出目录中指定扩展名的文件"""
        files: List[str] = []
        path = Path(directory)
        for ext in extensions:
            files.extend(str(p) for p in path.glob(f"*{ext}"))
        return sorted(files)

    @staticmethod
    def read_file(file_path: str) -> str:
        """读取纯文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def get_latest_run_dir(result_root: str, prefix: str) -> Optional[Path]:
        """获取指定前缀下最近一次运行目录"""
        root_path = Path(result_root)
        if not root_path.exists():
            return None
        run_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
