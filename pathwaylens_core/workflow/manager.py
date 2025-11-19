"""
WorkflowManager executes YAML/JSON-defined analysis workflows using existing
normalization and analysis engines in PathwayLens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from enum import Enum

import yaml
from loguru import logger

from ..normalization.normalizer import Normalizer
from ..analysis.engine import AnalysisEngine
from ..analysis.schemas import AnalysisParameters, AnalysisType
from ..comparison.engine import ComparisonEngine
from ..comparison.schemas import ComparisonParameters, ComparisonType
from ..reporting import generate_report


class WorkflowValidationError(Exception):
    pass


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    step_id: str
    type: str
    params: Dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class BatchJob:
    job_id: str
    workflow_spec: Dict[str, Any]
    status: WorkflowStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    error: Optional[str] = None


class WorkflowManager:
    """Load, validate, and execute workflows defined in YAML/JSON."""

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = Path(working_dir or ".").resolve()
        self.logger = logger.bind(module="workflow_manager")
        self.normalizer = Normalizer()
        self.analysis_engine = AnalysisEngine()
        # Simple in-memory artifact store mapping step_id -> artifact/result
        self.artifacts: Dict[str, Any] = {}
        self.comparison_engine = ComparisonEngine()
        
        # Batch processing
        self.job_queue: List[BatchJob] = []
        self.running_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.max_concurrent_jobs = 3
        self.scheduler_running = False

    def load(self, workflow_path: Path) -> Dict[str, Any]:
        path = Path(workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow file not found: {path}")
        text = path.read_text()
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(text) or {}
            return json.loads(text)
        except Exception as exc:
            raise WorkflowValidationError(f"Failed to parse workflow: {exc}")

    def validate(self, spec: Dict[str, Any]) -> List[WorkflowStep]:
        if not isinstance(spec, dict):
            raise WorkflowValidationError("Workflow spec must be a mapping")
        steps = spec.get("steps")
        if not steps or not isinstance(steps, list):
            raise WorkflowValidationError("Workflow must contain a non-empty 'steps' list")

        seen_ids = set()
        ordered_steps: List[WorkflowStep] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                raise WorkflowValidationError(f"Step {idx} must be a mapping")
            step_id = step.get("step_id")
            step_type = step.get("type")
            params = {k: v for k, v in step.items() if k not in {"step_id", "type"}}
            if not step_id or not isinstance(step_id, str):
                raise WorkflowValidationError(f"Step {idx} missing valid 'step_id'")
            if step_id in seen_ids:
                raise WorkflowValidationError(f"Duplicate step_id '{step_id}'")
            if step_type not in {"normalization", "analysis", "comparison"}:
                raise WorkflowValidationError(
                    f"Step '{step_id}' has unsupported type '{step_type}'"
                )
            seen_ids.add(step_id)
            ordered_steps.append(WorkflowStep(step_id=step_id, type=step_type, params=params))
        return ordered_steps

    async def run(self, steps: List[WorkflowStep]) -> Dict[str, Any]:
        workflow_id = str(uuid.uuid4())
        self.logger.info(f"Starting workflow {workflow_id} with {len(steps)} steps")
        results: Dict[str, Any] = {"workflow_id": workflow_id, "steps": []}

        for step in steps:
            if step.type == "normalization":
                result = await self._run_normalization(step)
            elif step.type == "analysis":
                result = await self._run_analysis(step)
            elif step.type == "comparison":
                result = await self._run_comparison(step)
            else:
                raise WorkflowValidationError(f"Unknown step type: {step.type}")

            results["steps"].append(result)
            self.artifacts[step.step_id] = result.get("artifact")

        # Optionally generate a simple report for the last step if requested
        if steps and steps[-1].params.get("report_dir"):
            report_dir = steps[-1].params.get("report_dir")
            last_result = results["steps"][-1].get("artifact")
            results["report_files"] = generate_report(last_result, report_dir, format=steps[-1].params.get("report_format", "html"))

        return results

    # Batch processing methods
    async def submit_batch_job(
        self, 
        workflow_spec: Dict[str, Any], 
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """Submit a workflow for batch processing."""
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            workflow_spec=workflow_spec,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            priority=priority,
            max_retries=max_retries
        )
        
        # Insert job in priority order
        inserted = False
        for i, existing_job in enumerate(self.job_queue):
            if priority > existing_job.priority:
                self.job_queue.insert(i, job)
                inserted = True
                break
        
        if not inserted:
            self.job_queue.append(job)
        
        self.logger.info(f"Submitted batch job {job_id} with priority {priority}")
        
        # Start scheduler if not running
        if not self.scheduler_running:
            asyncio.create_task(self._run_scheduler())
        
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get the status of a batch job."""
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            # Check queue
            for job in self.job_queue:
                if job.job_id == job_id:
                    return job
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        # Remove from queue
        self.job_queue = [job for job in self.job_queue if job.job_id != job_id]
        
        # Cancel running job
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
            job.status = WorkflowStatus.CANCELLED
            job.completed_at = datetime.now()
            self.completed_jobs[job_id] = job
            del self.running_jobs[job_id]
            return True
        
        return False

    async def _run_scheduler(self):
        """Run the batch job scheduler."""
        self.scheduler_running = True
        self.logger.info("Starting batch job scheduler")
        
        try:
            while True:
                # Check for completed jobs
                completed_job_ids = []
                for job_id, job in self.running_jobs.items():
                    if job.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                        completed_job_ids.append(job_id)
                
                # Move completed jobs
                for job_id in completed_job_ids:
                    job = self.running_jobs[job_id]
                    job.completed_at = datetime.now()
                    self.completed_jobs[job_id] = job
                    del self.running_jobs[job_id]
                
                # Start new jobs if we have capacity
                while (len(self.running_jobs) < self.max_concurrent_jobs and 
                       self.job_queue):
                    job = self.job_queue.pop(0)
                    asyncio.create_task(self._execute_batch_job(job))
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.scheduler_running = False

    async def _execute_batch_job(self, job: BatchJob):
        """Execute a single batch job."""
        job.status = WorkflowStatus.RUNNING
        job.started_at = datetime.now()
        self.running_jobs[job.job_id] = job
        
        try:
            # Validate and run workflow
            steps = self.validate(job.workflow_spec)
            result = await self.run(steps)
            
            job.status = WorkflowStatus.COMPLETED
            self.logger.info(f"Batch job {job.job_id} completed successfully")
            
        except Exception as e:
            job.error = str(e)
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                # Retry the job
                job.status = WorkflowStatus.PENDING
                job.started_at = None
                self.job_queue.append(job)
                self.logger.warning(f"Batch job {job.job_id} failed, retrying ({job.retry_count}/{job.max_retries})")
            else:
                job.status = WorkflowStatus.FAILED
                self.logger.error(f"Batch job {job.job_id} failed after {job.max_retries} retries: {e}")

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get the current status of the job queue."""
        return {
            "queue_size": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "max_concurrent": self.max_concurrent_jobs,
            "scheduler_running": self.scheduler_running
        }

    async def _run_normalization(self, step: WorkflowStep) -> Dict[str, Any]:
        params = step.params
        input_path = params.get("input")
        if not input_path:
            raise WorkflowValidationError(f"Normalization step '{step.step_id}' missing 'input'")
        species = params.get("species", "human")
        input_type = params.get("input_type", "auto")
        target_type = params.get("target_type", "symbol")
        target_species = params.get("target_species")
        ambiguity_policy = params.get("ambiguity_policy", "expand")

        # Delegate to Normalizer high-level API
        artifact = await self.normalizer.normalize_identifiers(
            input_data=input_path,
            input_format=input_type,
            output_format=target_type,
            species=species,
            target_species=target_species,
            ambiguity_policy=ambiguity_policy,
        )

        return {
            "step_id": step.step_id,
            "type": step.type,
            "status": "completed",
            "artifact": artifact,
        }

    async def _run_comparison(self, step: WorkflowStep) -> Dict[str, Any]:
        params = step.params
        inputs = params.get("inputs")
        if not inputs or not isinstance(inputs, list) or len(inputs) < 2:
            raise WorkflowValidationError(
                f"Comparison step '{step.step_id}' requires 'inputs' list with at least 2 references"
            )

        resolved_inputs = []
        for ref in inputs:
            if ref in self.artifacts:
                resolved_inputs.append(self.artifacts[ref])
            else:
                resolved_inputs.append(ref)

        comparison_type = params.get("comparison_type", "pathway_concordance").upper()
        species = params.get("species", "human")

        comp_params = ComparisonParameters(
            comparison_type=ComparisonType[comparison_type],
            species=species,
        )

        artifact = await self.comparison_engine.compare(
            analysis_results=resolved_inputs, parameters=comp_params
        )

        return {
            "step_id": step.step_id,
            "type": step.type,
            "status": "completed",
            "artifact": artifact,
        }

    async def _run_analysis(self, step: WorkflowStep) -> Dict[str, Any]:
        params = step.params
        input_ref = params.get("input")
        if not input_ref:
            raise WorkflowValidationError(f"Analysis step '{step.step_id}' missing 'input'")

        # Resolve input: either previous step_id or a path
        input_data = self.artifacts.get(input_ref, input_ref)

        method = params.get("method", "ORA").upper()
        databases = params.get("databases", ["kegg"]) or ["kegg"]
        species = params.get("species", "human")
        fdr = float(params.get("fdr", 0.05))

        analysis_type = AnalysisType.ORA if method == "ORA" else AnalysisType.GSEA
        analysis_params = AnalysisParameters(
            analysis_type=analysis_type,
            databases=databases,
            species=species,
            fdr_threshold=fdr,
        )

        artifact = await self.analysis_engine.analyze(
            input_data=input_data,
            parameters=analysis_params,
        )

        return {
            "step_id": step.step_id,
            "type": step.type,
            "status": "completed",
            "artifact": artifact,
        }


