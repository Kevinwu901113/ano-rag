"""Automatic calibration module for all parameters and thresholds.

This module implements automatic calibration of all weights and thresholds
using development data. It uses grid search or Bayesian optimization to
find optimal parameters and outputs them to calibration.json.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from loguru import logger
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, accuracy_score
import itertools
from datetime import datetime


class AutoCalibrator:
    """Automatic parameter calibration system."""
    
    def __init__(self, output_path: str = "calibration.json"):
        """
        Initialize the auto calibrator.
        
        Args:
            output_path: Path to save calibration results
        """
        self.output_path = output_path
        self.calibration_results = {}
        
        # Parameter grids for different components
        self.parameter_grids = {
            'learned_fusion': {
                'dense_weight': [0.8, 1.0, 1.2],
                'bm25_weight': [0.4, 0.6, 0.8],
                'title_weight': [0.2, 0.3, 0.4],
                'position_weight': [0.1, 0.2, 0.3]
            },
            'qa_coverage': {
                'threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
                'min_confidence': [0.1, 0.2, 0.3]
            },
            'structure_packing': {
                'max_sentences_per_paragraph': [1, 2, 3],
                'similarity_threshold': [0.2, 0.3, 0.4],
                'mmr_lambda': [0.6, 0.7, 0.8]
            },
            'span_picker': {
                'min_span_length': [1, 2],
                'max_span_length': [8, 10, 12],
                'confidence_threshold': [0.2, 0.3, 0.4]
            },
            'answer_verification': {
                'keep_threshold': [0.5, 0.6, 0.7, 0.8],
                'min_evidence_overlap': [0.05, 0.1, 0.15, 0.2]
            },
            'k_estimation': {
                'similarity_threshold': [0.2, 0.3, 0.4],
                'entity_overlap_threshold': [0.1, 0.2, 0.3],
                'min_k': [2],
                'max_k': [4, 5, 6]
            }
        }
    
    def calibrate_all_components(self, dev_data: List[Dict[str, Any]], 
                               evaluation_function: Callable) -> Dict[str, Any]:
        """
        Calibrate all components using development data.
        
        Args:
            dev_data: Development dataset for calibration
            evaluation_function: Function to evaluate system performance
                                Should take (dev_data, parameters) and return metrics
                                
        Returns:
            Calibration results
        """
        logger.info(f"Starting automatic calibration with {len(dev_data)} examples")
        
        # Initialize results
        self.calibration_results = {
            'calibration_date': datetime.now().isoformat(),
            'dev_data_size': len(dev_data),
            'components': {}
        }
        
        # Calibrate each component
        for component_name in self.parameter_grids.keys():
            logger.info(f"Calibrating {component_name}...")
            
            best_params = self._calibrate_component(
                component_name, dev_data, evaluation_function
            )
            
            self.calibration_results['components'][component_name] = best_params
        
        # Perform joint optimization (optional)
        logger.info("Performing joint optimization...")
        joint_results = self._joint_optimization(dev_data, evaluation_function)
        self.calibration_results['joint_optimization'] = joint_results
        
        # Save results
        self._save_calibration()
        
        logger.info("Calibration completed successfully")
        return self.calibration_results
    
    def _calibrate_component(self, component_name: str, dev_data: List[Dict[str, Any]], 
                           evaluation_function: Callable) -> Dict[str, Any]:
        """
        Calibrate a single component.
        
        Args:
            component_name: Name of the component to calibrate
            dev_data: Development data
            evaluation_function: Evaluation function
            
        Returns:
            Best parameters for the component
        """
        param_grid = self.parameter_grids.get(component_name, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {component_name}")
            return {}
        
        best_score = -np.inf
        best_params = {}
        best_metrics = {}
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations for {component_name}")
        
        for i, params in enumerate(param_combinations):
            try:
                # Create full parameter set (use defaults for other components)
                full_params = self._create_full_parameter_set(component_name, params)
                
                # Evaluate with these parameters
                metrics = evaluation_function(dev_data, full_params)
                
                # Use F1 score as primary metric (can be customized)
                score = metrics.get('f1', metrics.get('accuracy', 0.0))
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {e}")
                continue
        
        result = {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'search_space_size': len(param_combinations)
        }
        
        logger.info(f"Best {component_name} parameters: {best_params} (score: {best_score:.4f})")
        return result
    
    def _create_full_parameter_set(self, target_component: str, target_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a full parameter set with defaults for all components except the target.
        
        Args:
            target_component: Component being optimized
            target_params: Parameters for the target component
            
        Returns:
            Full parameter set
        """
        full_params = {}
        
        for component_name, param_grid in self.parameter_grids.items():
            if component_name == target_component:
                full_params[component_name] = target_params
            else:
                # Use default (middle) values for other components
                default_params = {}
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list) and param_values:
                        # Use middle value as default
                        middle_idx = len(param_values) // 2
                        default_params[param_name] = param_values[middle_idx]
                full_params[component_name] = default_params
        
        return full_params
    
    def _joint_optimization(self, dev_data: List[Dict[str, Any]], 
                          evaluation_function: Callable) -> Dict[str, Any]:
        """
        Perform joint optimization of critical parameters.
        
        Args:
            dev_data: Development data
            evaluation_function: Evaluation function
            
        Returns:
            Joint optimization results
        """
        # Select critical parameters for joint optimization
        critical_params = {
            'learned_fusion_dense_weight': [0.8, 1.0, 1.2],
            'learned_fusion_bm25_weight': [0.4, 0.6, 0.8],
            'qa_coverage_threshold': [0.4, 0.5, 0.6],
            'answer_verification_keep_threshold': [0.5, 0.6, 0.7]
        }
        
        best_score = -np.inf
        best_params = {}
        best_metrics = {}
        
        # Generate combinations (limit to reasonable size)
        param_combinations = list(ParameterGrid(critical_params))
        
        # Limit combinations if too many
        if len(param_combinations) > 100:
            # Sample randomly
            np.random.seed(42)
            indices = np.random.choice(len(param_combinations), 100, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        logger.info(f"Joint optimization with {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            try:
                # Convert to full parameter format
                full_params = self._convert_joint_params_to_full(params)
                
                # Evaluate
                metrics = evaluation_function(dev_data, full_params)
                score = metrics.get('f1', metrics.get('accuracy', 0.0))
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Joint optimization: {i + 1}/{len(param_combinations)} completed")
                    
            except Exception as e:
                logger.warning(f"Error in joint optimization with params {params}: {e}")
                continue
        
        result = {
            'best_joint_parameters': best_params,
            'best_joint_score': best_score,
            'best_joint_metrics': best_metrics,
            'combinations_tested': len(param_combinations)
        }
        
        logger.info(f"Best joint parameters: score {best_score:.4f}")
        return result
    
    def _convert_joint_params_to_full(self, joint_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert joint parameter format to full component format.
        
        Args:
            joint_params: Joint parameters with flattened names
            
        Returns:
            Full parameter set organized by component
        """
        full_params = {}
        
        # Initialize with defaults
        for component_name in self.parameter_grids.keys():
            full_params[component_name] = {}
        
        # Parse joint parameters
        for param_name, param_value in joint_params.items():
            if '_' in param_name:
                parts = param_name.split('_', 1)
                if len(parts) == 2:
                    component_name, actual_param_name = parts
                    if component_name in full_params:
                        full_params[component_name][actual_param_name] = param_value
        
        # Fill in defaults for missing parameters
        for component_name, param_grid in self.parameter_grids.items():
            for param_name, param_values in param_grid.items():
                if param_name not in full_params[component_name]:
                    if isinstance(param_values, list) and param_values:
                        middle_idx = len(param_values) // 2
                        full_params[component_name][param_name] = param_values[middle_idx]
        
        return full_params
    
    def calibrate_from_oracle_dump(self, oracle_dump_path: str, 
                                 evaluation_function: Callable) -> Dict[str, Any]:
        """
        Calibrate using oracle dump data.
        
        Args:
            oracle_dump_path: Path to oracle dump file
            evaluation_function: Evaluation function
            
        Returns:
            Calibration results
        """
        logger.info(f"Loading oracle dump from {oracle_dump_path}")
        
        try:
            with open(oracle_dump_path, 'r') as f:
                oracle_data = json.load(f)
            
            # Convert oracle data to dev format if needed
            dev_data = self._convert_oracle_to_dev_format(oracle_data)
            
            # Perform calibration
            return self.calibrate_all_components(dev_data, evaluation_function)
            
        except Exception as e:
            logger.error(f"Failed to load oracle dump: {e}")
            raise
    
    def _convert_oracle_to_dev_format(self, oracle_data: Any) -> List[Dict[str, Any]]:
        """
        Convert oracle dump format to development data format.
        
        Args:
            oracle_data: Oracle dump data
            
        Returns:
            Development data in standard format
        """
        # This method should be customized based on the actual oracle dump format
        dev_data = []
        
        if isinstance(oracle_data, list):
            for item in oracle_data:
                if isinstance(item, dict):
                    # Extract relevant fields
                    dev_item = {
                        'question': item.get('question', ''),
                        'gold_answer': item.get('answer', ''),
                        'passages': item.get('passages', []),
                        'support_idxs': item.get('support_idxs', []),
                        # Add other relevant fields
                    }
                    dev_data.append(dev_item)
        
        logger.info(f"Converted oracle dump to {len(dev_data)} development examples")
        return dev_data
    
    def _save_calibration(self) -> None:
        """
        Save calibration results to file.
        """
        try:
            with open(self.output_path, 'w') as f:
                json.dump(self.calibration_results, f, indent=2)
            
            logger.info(f"Calibration results saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration results: {e}")
            raise
    
    def load_calibration(self, path: str) -> Dict[str, Any]:
        """
        Load existing calibration results.
        
        Args:
            path: Path to calibration file
            
        Returns:
            Calibration results
        """
        try:
            with open(path, 'r') as f:
                calibration = json.load(f)
            
            logger.info(f"Calibration loaded from {path}")
            return calibration
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise
    
    def get_oracle_upper_bound(self, dev_data: List[Dict[str, Any]], 
                             evaluation_function: Callable) -> Dict[str, float]:
        """
        Compute oracle upper bound performance.
        
        Args:
            dev_data: Development data
            evaluation_function: Evaluation function
            
        Returns:
            Oracle performance metrics
        """
        logger.info("Computing oracle upper bound performance")
        
        # Create oracle parameters (perfect settings)
        oracle_params = {
            'learned_fusion': {'use_oracle': True},
            'qa_coverage': {'use_oracle': True},
            'structure_packing': {'use_oracle': True},
            'span_picker': {'use_oracle': True},
            'answer_verification': {'use_oracle': True},
            'k_estimation': {'use_oracle': True}
        }
        
        try:
            oracle_metrics = evaluation_function(dev_data, oracle_params)
            
            logger.info(f"Oracle upper bound: {oracle_metrics}")
            return oracle_metrics
            
        except Exception as e:
            logger.warning(f"Failed to compute oracle upper bound: {e}")
            return {}
    
    def analyze_parameter_sensitivity(self, dev_data: List[Dict[str, Any]], 
                                    evaluation_function: Callable, 
                                    component_name: str) -> Dict[str, Any]:
        """
        Analyze parameter sensitivity for a specific component.
        
        Args:
            dev_data: Development data
            evaluation_function: Evaluation function
            component_name: Component to analyze
            
        Returns:
            Sensitivity analysis results
        """
        logger.info(f"Analyzing parameter sensitivity for {component_name}")
        
        param_grid = self.parameter_grids.get(component_name, {})
        sensitivity_results = {}
        
        for param_name, param_values in param_grid.items():
            param_scores = []
            
            for param_value in param_values:
                # Create parameter set with only this parameter varied
                params = {param_name: param_value}
                full_params = self._create_full_parameter_set(component_name, params)
                
                try:
                    metrics = evaluation_function(dev_data, full_params)
                    score = metrics.get('f1', metrics.get('accuracy', 0.0))
                    param_scores.append(score)
                except Exception as e:
                    logger.warning(f"Error evaluating {param_name}={param_value}: {e}")
                    param_scores.append(0.0)
            
            # Compute sensitivity metrics
            if param_scores:
                sensitivity_results[param_name] = {
                    'values': param_values,
                    'scores': param_scores,
                    'range': max(param_scores) - min(param_scores),
                    'std': np.std(param_scores),
                    'best_value': param_values[np.argmax(param_scores)],
                    'best_score': max(param_scores)
                }
        
        logger.info(f"Sensitivity analysis completed for {component_name}")
        return sensitivity_results


def create_auto_calibrator(output_path: str = "calibration.json") -> AutoCalibrator:
    """
    Create an AutoCalibrator instance.
    
    Args:
        output_path: Path to save calibration results
        
    Returns:
        AutoCalibrator instance
    """
    return AutoCalibrator(output_path=output_path)


# Example evaluation function template
def example_evaluation_function(dev_data: List[Dict[str, Any]], 
                              parameters: Dict[str, Any]) -> Dict[str, float]:
    """
    Example evaluation function template.
    
    Args:
        dev_data: Development data
        parameters: Parameters to evaluate
        
    Returns:
        Evaluation metrics
    """
    # This should be implemented based on the actual system
    # It should:
    # 1. Configure the system with the given parameters
    # 2. Run the system on dev_data
    # 3. Compute and return metrics (f1, accuracy, etc.)
    
    # Placeholder implementation
    return {
        'f1': 0.5,
        'accuracy': 0.6,
        'precision': 0.55,
        'recall': 0.45
    }