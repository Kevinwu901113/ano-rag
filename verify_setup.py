#!/usr/bin/env python3
"""
验证批量测试环境设置
检查所有必要的文件和依赖是否就绪
"""

import os
import sys
import json
from pathlib import Path
from loguru import logger

def check_file_exists(file_path: Path, description: str) -> bool:
    """检查文件是否存在"""
    if file_path.exists():
        logger.info(f"✓ {description}: {file_path}")
        return True
    else:
        logger.error(f"✗ {description} not found: {file_path}")
        return False

def check_directory_exists(dir_path: Path, description: str) -> bool:
    """检查目录是否存在"""
    if dir_path.exists() and dir_path.is_dir():
        logger.info(f"✓ {description}: {dir_path}")
        return True
    else:
        logger.warning(f"⚠ {description} not found: {dir_path}")
        return False

def check_json_format(file_path: Path, description: str) -> bool:
    """检查JSON文件格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                # 检查JSONL格式
                line_count = 0
                for line in f:
                    if line.strip():
                        json.loads(line)
                        line_count += 1
                logger.info(f"✓ {description} format valid ({line_count} lines)")
            else:
                # 检查JSON格式
                json.load(f)
                logger.info(f"✓ {description} format valid")
        return True
    except Exception as e:
        logger.error(f"✗ {description} format error: {e}")
        return False

def check_python_imports() -> bool:
    """检查Python依赖"""
    required_modules = [
        'loguru', 'pathlib', 'json', 'subprocess', 'tempfile', 'shutil'
    ]
    
    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ Python module: {module}")
        except ImportError:
            logger.error(f"✗ Python module missing: {module}")
            all_good = False
    
    return all_good

def main():
    logger.info("验证批量测试环境设置")
    logger.info("=" * 50)
    
    project_root = Path(__file__).parent
    all_checks_passed = True
    
    # 检查核心脚本文件
    logger.info("\n检查脚本文件:")
    script_files = [
        (project_root / "main.py", "主程序"),
        (project_root / "batch_test.py", "批量测试脚本"),
        (project_root / "test_batch.py", "测试脚本"),
        (project_root / "run_musique_batch.py", "Musique批量处理脚本"),
        (project_root / "verify_setup.py", "验证脚本")
    ]
    
    for file_path, description in script_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # 检查配置文件
    logger.info("\n检查配置文件:")
    config_files = [
        (project_root / "config.yaml", "配置文件"),
        (project_root / "requirements.txt", "依赖文件")
    ]
    
    for file_path, description in config_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # 检查示例数据文件
    logger.info("\n检查数据文件:")
    data_files = [
        (project_root / "example.jsonl", "示例数据文件"),
        (project_root / "query.json", "查询结果示例")
    ]
    
    for file_path, description in data_files:
        if check_file_exists(file_path, description):
            check_json_format(file_path, description)
        else:
            all_checks_passed = False
    
    # 检查数据目录
    logger.info("\n检查目录结构:")
    directories = [
        (project_root / "data", "数据目录"),
        (project_root / "result", "结果目录")
    ]
    
    for dir_path, description in directories:
        check_directory_exists(dir_path, description)
    
    # 检查musique数据文件（可选）
    musique_file = project_root / "data" / "musique_ans_v1.0_train.jsonl"
    if check_file_exists(musique_file, "Musique训练数据"):
        logger.info("  检查前几行格式...")
        try:
            with open(musique_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:  # 只检查前3行
                        break
                    if line.strip():
                        data = json.loads(line)
                        if 'id' in data and 'question' in data and 'paragraphs' in data:
                            logger.info(f"  ✓ Line {i+1} format correct")
                        else:
                            logger.warning(f"  ⚠ Line {i+1} missing required fields")
        except Exception as e:
            logger.error(f"  ✗ Musique file format error: {e}")
    else:
        logger.warning("  Musique数据文件不存在，将无法运行完整批量处理")
    
    # 检查Python依赖
    logger.info("\n检查Python依赖:")
    if not check_python_imports():
        all_checks_passed = False
    
    # 检查项目模块
    logger.info("\n检查项目模块:")
    project_modules = ['config', 'doc', 'query', 'utils']
    for module in project_modules:
        module_path = project_root / module
        if check_directory_exists(module_path, f"{module}模块目录"):
            init_file = module_path / "__init__.py"
            check_file_exists(init_file, f"{module}模块初始化文件")
        else:
            all_checks_passed = False
    
    # 总结
    logger.info("\n" + "=" * 50)
    if all_checks_passed:
        logger.info("✓ 所有检查通过！环境设置正确。")
        logger.info("\n可以开始使用批量测试脚本:")
        logger.info("1. 测试运行: python test_batch.py")
        logger.info("2. 小批量处理: python run_musique_batch.py --limit 5")
        logger.info("3. 完整处理: python run_musique_batch.py")
    else:
        logger.error("✗ 部分检查失败，请修复上述问题后重试。")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())