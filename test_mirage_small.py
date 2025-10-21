#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-scale test for main_mirage.py functionality

This script creates a minimal test dataset to verify that main_mirage.py
works correctly and produces the expected output format.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

def create_test_data():
    """Create minimal test data for MIRAGE"""
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="mirage_test_"))
    mirage_dir = test_dir / "mirage"
    mirage_dir.mkdir()
    
    # Create minimal dataset.json
    dataset = [
        {
            "source": "test",
            "query_id": "test-001",
            "query": "What is the capital of France?",
            "doc_name": "France",
            "answer": ["Paris"],
            "doc_url": "https://example.com/france",
            "num_doc_labels": 1
        },
        {
            "source": "test",
            "query_id": "test-002", 
            "query": "Who wrote Romeo and Juliet?",
            "doc_name": "Shakespeare",
            "answer": ["William Shakespeare", "Shakespeare"],
            "doc_url": "https://example.com/shakespeare",
            "num_doc_labels": 1
        }
    ]
    
    # Create minimal doc_pool.json
    doc_pool = [
        {
            "mapped_id": "test-001",
            "doc_name": "France",
            "doc_chunk": "France is a country in Western Europe. The capital and largest city of France is Paris, which is located in the north-central part of the country.",
            "support": 1
        },
        {
            "mapped_id": "test-001", 
            "doc_name": "France",
            "doc_chunk": "France has a rich history and culture. It is known for its cuisine, art, and architecture. The French language is spoken by millions of people worldwide.",
            "support": 0
        },
        {
            "mapped_id": "test-002",
            "doc_name": "Shakespeare",
            "doc_chunk": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth.",
            "support": 1
        },
        {
            "mapped_id": "test-002",
            "doc_name": "Shakespeare", 
            "doc_chunk": "Shakespeare lived from 1564 to 1616. He is widely regarded as the greatest writer in the English language and the world's greatest dramatist.",
            "support": 0
        }
    ]
    
    # Create minimal oracle.json
    oracle = {
        "test-001": {
            "mapped_id": "test-001",
            "doc_name": "France", 
            "doc_chunk": "France is a country in Western Europe. The capital and largest city of France is Paris, which is located in the north-central part of the country.",
            "support": 1
        },
        "test-002": {
            "mapped_id": "test-002",
            "doc_name": "Shakespeare",
            "doc_chunk": "William Shakespeare was an English playwright and poet. He wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth.",
            "support": 1
        }
    }
    
    # Write files
    with open(mirage_dir / "dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    with open(mirage_dir / "doc_pool.json", 'w') as f:
        json.dump(doc_pool, f, indent=2)
        
    with open(mirage_dir / "oracle.json", 'w') as f:
        json.dump(oracle, f, indent=2)
    
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