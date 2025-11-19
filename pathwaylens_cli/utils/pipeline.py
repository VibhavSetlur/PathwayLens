"""
Pipeline execution utilities for CLI.

Chains multiple operations (normalize → analyze → compare) in a single command.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from dataclasses import dataclass


@dataclass
class PipelineStep:
    """A step in a pipeline."""
    name: str
    func: Callable
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any]


class PipelineExecutor:
    """Execute analysis pipelines."""
    
    def __init__(self):
        """Initialize pipeline executor."""
        self.logger = logger.bind(module="pipeline_executor")
        self.steps: List[PipelineStep] = []
        self.artifacts: Dict[str, Any] = {}
    
    def add_step(
        self,
        name: str,
        func: Callable,
        inputs: List[str],
        outputs: List[str],
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Add a step to the pipeline.
        
        Args:
            name: Step name
            func: Function to execute
            inputs: List of input artifact names
            outputs: List of output artifact names
            params: Additional parameters
        """
        step = PipelineStep(
            name=name,
            func=func,
            inputs=inputs,
            outputs=outputs,
            params=params or {}
        )
        self.steps.append(step)
        self.logger.debug(f"Added pipeline step: {name}")
    
    async def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            initial_inputs: Initial input artifacts
            
        Returns:
            Final artifacts
        """
        self.logger.info(f"Executing pipeline with {len(self.steps)} steps")
        
        # Initialize artifacts with initial inputs
        self.artifacts.update(initial_inputs)
        
        # Execute each step
        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                # Get inputs for this step
                step_inputs = {
                    inp: self.artifacts[inp]
                    for inp in step.inputs
                    if inp in self.artifacts
                }
                
                # Add step parameters
                step_inputs.update(step.params)
                
                # Execute step
                if asyncio.iscoroutinefunction(step.func):
                    step_outputs = await step.func(**step_inputs)
                else:
                    step_outputs = step.func(**step_inputs)
                
                # Store outputs
                if isinstance(step_outputs, dict):
                    for output_name in step.outputs:
                        if output_name in step_outputs:
                            self.artifacts[output_name] = step_outputs[output_name]
                elif len(step.outputs) == 1:
                    # Single output
                    self.artifacts[step.outputs[0]] = step_outputs
                else:
                    # Multiple outputs - assume dict
                    if isinstance(step_outputs, dict):
                        self.artifacts.update(step_outputs)
                
                self.logger.info(f"Step {step.name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {e}")
                raise
        
        self.logger.info("Pipeline execution completed")
        return self.artifacts
    
    def execute_sync(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of execute."""
        return asyncio.run(self.execute(initial_inputs))
    
    def reset(self):
        """Reset pipeline state."""
        self.steps = []
        self.artifacts = {}


def create_analysis_pipeline(
    input_file: Path,
    normalize_params: Dict[str, Any],
    analyze_params: Dict[str, Any]
) -> PipelineExecutor:
    """
    Create a standard normalize → analyze pipeline.
    
    Args:
        input_file: Input file path
        normalize_params: Normalization parameters
        analyze_params: Analysis parameters
        
    Returns:
        Configured PipelineExecutor
    """
    pipeline = PipelineExecutor()
    
    # Step 1: Normalize
    async def normalize_step(input_file: Path, **params):
        from pathwaylens_core.normalization import Normalizer
        normalizer = Normalizer()
        # Would call normalizer here
        return {'normalized_data': 'placeholder'}
    
    pipeline.add_step(
        name='normalize',
        func=lambda **kwargs: normalize_step(input_file, **kwargs),
        inputs=['input_file'],
        outputs=['normalized_data'],
        params=normalize_params
    )
    
    # Step 2: Analyze
    async def analyze_step(normalized_data: Any, **params):
        from pathwaylens_core.analysis import AnalysisEngine
        engine = AnalysisEngine()
        # Would call engine here
        return {'analysis_results': 'placeholder'}
    
    pipeline.add_step(
        name='analyze',
        func=lambda **kwargs: analyze_step(**kwargs),
        inputs=['normalized_data'],
        outputs=['analysis_results'],
        params=analyze_params
    )
    
    return pipeline



