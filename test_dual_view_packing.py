#!/usr/bin/env python3
"""Test script for dual view packing strategy."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context.packer import ContextPacker
from loguru import logger
import yaml

def load_test_config():
    """Load test configuration with dual view packing enabled."""
    config = {
        'context': {
            'dual_view_packing': {
                'enabled': True,
                'facts_ratio': 0.7,
                'original_ratio': 0.3,
                'fact_selection': {
                    'min_score': 0.3,
                    'min_coverage': 0.2,
                    'max_facts': 15,
                    'diversity_weight': 0.4,
                    'entity_diversity': True,
                    'predicate_diversity': True,
                    'temporal_diversity': True
                },
                'span_alignment': {
                    'max_spans': 8,
                    'min_span_length': 20,
                    'max_span_length': 150,
                    'overlap_threshold': 0.8,
                    'verification_priority': True
                }
            }
        }
    }
    return config

def create_test_notes():
    """Create test notes with atomic facts structure."""
    notes = [
        {
            'note_id': 'note_1',
            'content': 'Albert Einstein was born in Germany in 1879. He developed the theory of relativity.',
            'similarity': 0.9,
            'atomic_facts': [
                {
                    'fact': 'Albert Einstein was born in Germany',
                    'importance': 0.8,
                    'entities': ['Albert Einstein', 'Germany'],
                    'predicates': ['was born'],
                    'temporal': ['1879'],
                    'span': {'start': 0, 'end': 42}
                },
                {
                    'fact': 'Albert Einstein developed the theory of relativity',
                    'importance': 0.9,
                    'entities': ['Albert Einstein', 'theory of relativity'],
                    'predicates': ['developed'],
                    'temporal': [],
                    'span': {'start': 52, 'end': 95}
                }
            ]
        },
        {
            'note_id': 'note_2',
            'content': 'The theory of relativity revolutionized physics. It includes special and general relativity.',
            'similarity': 0.85,
            'atomic_facts': [
                {
                    'fact': 'The theory of relativity revolutionized physics',
                    'importance': 0.85,
                    'entities': ['theory of relativity', 'physics'],
                    'predicates': ['revolutionized'],
                    'temporal': [],
                    'span': {'start': 0, 'end': 47}
                },
                {
                    'fact': 'Theory of relativity includes special and general relativity',
                    'importance': 0.7,
                    'entities': ['theory of relativity', 'special relativity', 'general relativity'],
                    'predicates': ['includes'],
                    'temporal': [],
                    'span': {'start': 49, 'end': 92}
                }
            ]
        },
        {
            'note_id': 'note_3',
            'content': 'Einstein won the Nobel Prize in Physics in 1921 for his work on photoelectric effect.',
            'similarity': 0.75,
            'atomic_facts': [
                {
                    'fact': 'Einstein won the Nobel Prize in Physics in 1921',
                    'importance': 0.8,
                    'entities': ['Einstein', 'Nobel Prize', 'Physics'],
                    'predicates': ['won'],
                    'temporal': ['1921'],
                    'span': {'start': 0, 'end': 50}
                },
                {
                    'fact': 'Einstein won Nobel Prize for work on photoelectric effect',
                    'importance': 0.75,
                    'entities': ['Einstein', 'Nobel Prize', 'photoelectric effect'],
                    'predicates': ['won', 'work on'],
                    'temporal': [],
                    'span': {'start': 0, 'end': 86}
                }
            ]
        }
    ]
    return notes

def test_dual_view_packing():
    """Test the dual view packing functionality."""
    logger.info("Starting dual view packing test")
    
    # Load test configuration
    config = load_test_config()
    
    # Create context packer with test config
    packer = ContextPacker(config=config)
    
    # Create test notes
    notes = create_test_notes()
    
    # Test question
    question = "What did Einstein contribute to physics?"
    
    # Pack context using dual view strategy
    try:
        prompt, passages_by_idx, packed_order = packer.pack_context(
            notes=notes,
            question=question,
            token_budget=2000
        )
        
        logger.info("Dual view packing successful!")
        logger.info(f"Generated prompt length: {len(prompt)} characters")
        logger.info(f"Number of passages: {len(passages_by_idx)}")
        logger.info(f"Packed order: {packed_order}")
        
        print("\n" + "="*80)
        print("GENERATED PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80)
        
        print("\nPASSAGES BY INDEX:")
        for idx, passage in passages_by_idx.items():
            print(f"\n{idx}: {passage}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dual view packing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_to_regular_packing():
    """Test fallback when dual view packing is disabled."""
    logger.info("Testing fallback to regular packing")
    
    # Create config with dual view disabled
    config = {
        'context': {
            'dual_view_packing': {
                'enabled': False
            }
        }
    }
    
    packer = ContextPacker(config=config)
    notes = create_test_notes()
    question = "What did Einstein contribute to physics?"
    
    try:
        prompt, passages_by_idx, packed_order = packer.pack_context(
            notes=notes,
            question=question,
            token_budget=2000
        )
        
        logger.info("Fallback to regular packing successful!")
        logger.info(f"Generated prompt length: {len(prompt)} characters")
        return True
        
    except Exception as e:
        logger.error(f"Fallback packing failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Running dual view packing tests")
    
    # Test 1: Dual view packing
    success1 = test_dual_view_packing()
    
    # Test 2: Fallback to regular packing
    success2 = test_fallback_to_regular_packing()
    
    if success1 and success2:
        logger.info("All tests passed!")
        print("\n✅ All dual view packing tests passed!")
    else:
        logger.error("Some tests failed!")
        print("\n❌ Some tests failed!")
        sys.exit(1)