#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-scale test for main_mirage.py functionality

This script copies the bundled MIRAGE small dataset fixtures to verify that
main_mirage.py works correctly and produces the expected output format.
"""

import json
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create test data for MIRAGE using the bundled small dataset"""
    test_dir = Path(tempfile.mkdtemp(prefix="mirage_test_"))
    mirage_dir = test_dir / "mirage"
    mirage_dir.mkdir()

    repo_mirage_dir = Path(__file__).resolve().parent / "MIRAGE" / "mirage"
    source_to_target = {
        "dataset_small.json": "dataset.json",
        "doc_pool_small.json": "doc_pool.json",
        "oracle_small.json": "oracle.json",
    }

    for source_name, target_name in source_to_target.items():
        source_path = repo_mirage_dir / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Required test fixture not found: {source_path}")
        shutil.copy(source_path, mirage_dir / target_name)

    print(f"Test data created in: {test_dir}")
    return test_dir

def validate_output(result_dir: Path):
    """Validate the output format"""
    predictions_file = result_dir / "predictions.jsonl"
    manifest_file = result_dir / "manifest.json"
    
    if not predictions_file.exists():
        print("❌ predictions.jsonl not found")
        return False
    
    if not manifest_file.exists():
        print("❌ manifest.json not found")
        return False
    
    # Validate predictions.jsonl format
    try:
        predictions = []
        with open(predictions_file, 'r') as f:
            for line in f:
                pred = json.loads(line.strip())
                predictions.append(pred)

        print(f"✅ Found {len(predictions)} predictions")

        # Check required fields
        for i, pred in enumerate(predictions):
            required_fields = ['id', 'predicted_answer', 'retrieved_contexts']
            for field in required_fields:
                if field not in pred:
                    print(f"❌ Missing field '{field}' in prediction {i}")
                    return False

            # Check retrieved_contexts format
            for j, ctx in enumerate(pred['retrieved_contexts']):
                if 'title' not in ctx or 'text' not in ctx:
                    print(f"❌ Invalid context format in prediction {i}, context {j}")
                    return False

        print("✅ Predictions format is valid")

        # Validate manifest metadata
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)

        data_info = manifest.get('data_info', {})
        required_data_fields = [
            'dataset_path',
            'dataset_size',
            'dataset_hash',
            'dataset_file_size_bytes',
            'dataset_line_count',
            'doc_pool_path',
            'doc_pool_size',
            'doc_pool_hash',
            'doc_pool_file_size_bytes',
            'doc_pool_line_count'
        ]

        for field in required_data_fields:
            if field not in data_info:
                print(f"❌ Missing field '{field}' in manifest data_info")
                return False

        if data_info.get('oracle_path'):
            oracle_fields = [
                'oracle_hash',
                'oracle_file_size_bytes',
                'oracle_line_count'
            ]
            for field in oracle_fields:
                if field not in data_info:
                    print(f"❌ Missing field '{field}' in manifest oracle data")
                    return False

        logs_files = manifest.get('output_files', {}).get('logs', [])
        if logs_files != ['logs/run.log', 'logs/run_error.log']:
            print(f"❌ Unexpected log file entries in manifest: {logs_files}")
            return False

        print("✅ Manifest metadata is complete")

        # Print sample prediction
        if predictions:
            print("\n📋 Sample prediction:")
            sample = predictions[0]
            print(f"ID: {sample['id']}")
            print(f"Answer: {sample['predicted_answer']}")
            print(f"Contexts: {len(sample['retrieved_contexts'])}")
            for i, ctx in enumerate(sample['retrieved_contexts']):
                print(f"  Context {i+1}: {ctx['title']} - {ctx['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating predictions: {e}")
        return False

def main():
    """Run the test"""
    print("🧪 Creating test data...")
    test_dir = create_test_data()
    
    try:
        # Test different modes
        modes = ['base', 'oracle', 'mixed']
        
        for mode in modes:
            print(f"\n🔄 Testing {mode} mode...")
            
            # Run main_mirage.py
            cmd = [
                'python', 'main_mirage.py',
                '--mode', mode,
                '--topk', '2',
                '--dataset', str(test_dir / 'mirage' / 'dataset.json'),
                '--doc_pool', str(test_dir / 'mirage' / 'doc_pool.json'),
                '--oracle', str(test_dir / 'mirage' / 'oracle.json'),
                '--result_dir', str(test_dir / 'result'),
                '--retriever', 'bm25',  # Use BM25 for simplicity
                '--model', 'test-model',
                '--temperature', '0.1',
                '--max_tokens', '100',
                '--new',
                '--debug'
            ]
            
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                print(f"❌ {mode} mode failed:")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                continue
            
            # Find the result directory
            result_base = test_dir / 'result'
            if result_base.exists():
                run_dirs = [d for d in result_base.iterdir() if d.is_dir()]
                if run_dirs:
                    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                    print(f"✅ {mode} mode completed: {latest_run}")
                    
                    # Validate output
                    if validate_output(latest_run):
                        print(f"✅ {mode} mode output is valid")
                    else:
                        print(f"❌ {mode} mode output is invalid")
                else:
                    print(f"❌ No run directories found for {mode} mode")
            else:
                print(f"❌ Result directory not found for {mode} mode")
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test data: {test_dir}")
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()
