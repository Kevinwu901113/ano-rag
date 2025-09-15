#!/usr/bin/env python3
"""Test script for enhanced graph retrieval with path planning and dynamic thresholds."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import List, Dict, Any
from loguru import logger

# Import the enhanced components
from graph.graph_builder import GraphBuilder
from graph.graph_index import GraphIndex
from graph.graph_retriever import GraphRetriever
from graph.enhanced_path_planner import EnhancedPathPlanner
from graph.dynamic_threshold_adjuster import DynamicThresholdAdjuster


def create_test_atomic_notes() -> List[Dict[str, Any]]:
    """Create test atomic notes with entities, relations, and temporal information."""
    return [
        {
            'id': 'note_1',
            'content': 'Apple Inc. was founded by Steve Jobs in 1976. The company revolutionized personal computing.',
            'entities': ['Apple Inc.', 'Steve Jobs', 'personal computing'],
            'relations': [('Steve Jobs', 'founded', 'Apple Inc.'), ('Apple Inc.', 'revolutionized', 'personal computing')],
            'temporal_info': ['1976'],
            'embedding': np.random.rand(384).tolist()
        },
        {
            'id': 'note_2', 
            'content': 'Steve Jobs returned to Apple in 1997 and launched the iPhone in 2007.',
            'entities': ['Steve Jobs', 'Apple', 'iPhone'],
            'relations': [('Steve Jobs', 'returned_to', 'Apple'), ('Apple', 'launched', 'iPhone')],
            'temporal_info': ['1997', '2007'],
            'embedding': np.random.rand(384).tolist()
        },
        {
            'id': 'note_3',
            'content': 'The iPhone transformed the smartphone industry and mobile computing.',
            'entities': ['iPhone', 'smartphone industry', 'mobile computing'],
            'relations': [('iPhone', 'transformed', 'smartphone industry'), ('iPhone', 'enabled', 'mobile computing')],
            'temporal_info': [],
            'embedding': np.random.rand(384).tolist()
        },
        {
            'id': 'note_4',
            'content': 'Microsoft was founded by Bill Gates and Paul Allen in 1975.',
            'entities': ['Microsoft', 'Bill Gates', 'Paul Allen'],
            'relations': [('Bill Gates', 'founded', 'Microsoft'), ('Paul Allen', 'founded', 'Microsoft')],
            'temporal_info': ['1975'],
            'embedding': np.random.rand(384).tolist()
        },
        {
            'id': 'note_5',
            'content': 'Google was founded by Larry Page and Sergey Brin in 1998. They developed the PageRank algorithm.',
            'entities': ['Google', 'Larry Page', 'Sergey Brin', 'PageRank algorithm'],
            'relations': [('Larry Page', 'founded', 'Google'), ('Sergey Brin', 'founded', 'Google'), 
                         ('Larry Page', 'developed', 'PageRank algorithm'), ('Sergey Brin', 'developed', 'PageRank algorithm')],
            'temporal_info': ['1998'],
            'embedding': np.random.rand(384).tolist()
        }
    ]


def test_enhanced_path_planner():
    """Test the enhanced path planner functionality."""
    logger.info("Testing Enhanced Path Planner...")
    
    atomic_notes = create_test_atomic_notes()
    
    try:
        # Build graph and graph index first
        builder = GraphBuilder()
        embeddings = np.array([note['embedding'] for note in atomic_notes])
        graph = builder.build_graph(atomic_notes, embeddings)
        
        # Create graph index
        graph_index = GraphIndex(graph)
        graph_index.build_index(graph, atomic_notes, embeddings)
        
        # Initialize the enhanced path planner
        planner = EnhancedPathPlanner(graph_index, atomic_notes)
        logger.info("Enhanced path planner initialized successfully")
        
        # Test path planning
        query_entities = ['Steve Jobs', 'iPhone']
        query_temporal = ['2007']
        initial_candidates = ['note_1', 'note_2', 'note_3']
        
        paths = planner.plan_reasoning_paths(query_entities, query_temporal, initial_candidates)
        
        logger.info(f"Found {len(paths)} reasoning paths")
        for i, path in enumerate(paths[:3]):  # Show top 3 paths
            logger.info(f"Path {i+1}: score={path['score']:.3f}, path={path['path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced path planner test failed: {e}")
        return False


def test_dynamic_threshold_adjuster():
    """Test the dynamic threshold adjuster functionality."""
    logger.info("Testing Dynamic Threshold Adjuster...")
    
    atomic_notes = create_test_atomic_notes()
    
    try:
        # Initialize the dynamic threshold adjuster
        adjuster = DynamicThresholdAdjuster(atomic_notes)
        logger.info("Dynamic threshold adjuster initialized successfully")
        
        # Test quality analysis
        quality_insights = adjuster.get_quality_insights()
        logger.info(f"Quality insights: {quality_insights}")
        
        # Test adaptive thresholds
        query = "When did Steve Jobs launch the iPhone?"
        candidate_notes = atomic_notes[:3]
        
        thresholds = adjuster.get_adaptive_thresholds(
            query, candidate_notes,
            base_similarity_threshold=0.7,
            base_path_score_threshold=0.5
        )
        
        logger.info(f"Adaptive thresholds: {thresholds}")
        
        # Test retrieval parameter adjustment
        adjusted_params = adjuster.adjust_retrieval_params(
            query_keywords=['Steve', 'Jobs', 'iPhone', 'launch'],
            query_entities=['Steve Jobs', 'iPhone'],
            current_top_k=10
        )
        
        logger.info(f"Adjusted retrieval params: {adjusted_params}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dynamic threshold adjuster test failed: {e}")
        return False


def test_integrated_graph_retriever():
    """Test the integrated graph retriever with enhanced features."""
    logger.info("Testing Integrated Graph Retriever...")
    
    atomic_notes = create_test_atomic_notes()
    
    try:
        # Build graph
        builder = GraphBuilder()
        embeddings = np.array([note['embedding'] for note in atomic_notes])
        graph = builder.build_graph(atomic_notes, embeddings)
        
        # Create graph index
        graph_index = GraphIndex(graph)
        graph_index.build_index(graph, atomic_notes, embeddings)
        
        # Initialize enhanced graph retriever
        retriever = GraphRetriever(graph_index, k_hop=2, atomic_notes=atomic_notes)
        logger.info("Enhanced graph retriever initialized successfully")
        
        # Test retrieval with reasoning paths
        query_embedding = np.random.rand(384)
        query_keywords = ['Steve', 'Jobs', 'iPhone', 'launch']
        query_entities = ['Steve Jobs', 'iPhone']
        
        results = retriever.retrieve_with_reasoning_paths(
            query_embedding,
            top_k=5,
            query_keywords=query_keywords,
            query_entities=query_entities
        )
        
        logger.info(f"Retrieved {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: id={result.get('id', 'unknown')}, score={result.get('graph_score', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integrated graph retriever test failed: {e}")
        return False


def test_import_integrity():
    """Test that all imports work correctly."""
    logger.info("Testing Import Integrity...")
    
    try:
        # Test all key imports
        from graph.enhanced_path_planner import EnhancedPathPlanner
        from graph.dynamic_threshold_adjuster import DynamicThresholdAdjuster
        from graph.graph_retriever import GraphRetriever
        from graph.graph_index import GraphIndex
        from graph.graph_builder import GraphBuilder
        
        logger.info("All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"Import test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in import test: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting Enhanced Graph Retrieval Tests")
    
    tests = [
        ("Import Integrity", test_import_integrity),
        ("Enhanced Path Planner", test_enhanced_path_planner),
        ("Dynamic Threshold Adjuster", test_dynamic_threshold_adjuster),
        ("Integrated Graph Retriever", test_integrated_graph_retriever)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Test Summary")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced graph retrieval is working correctly.")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)