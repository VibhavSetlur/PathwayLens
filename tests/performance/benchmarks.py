"""
Performance benchmarking utilities and baseline metrics.

This module provides utilities for running performance benchmarks
and tracking performance regressions.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    test_name: str
    operation: str
    elapsed_time: float
    memory_usage: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for regression testing."""
    test_name: str
    max_elapsed_time: float
    max_memory_usage: Optional[float] = None
    min_throughput: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceBenchmark:
    """Utility class for running and tracking performance benchmarks."""
    
    def __init__(self, baseline_file: Optional[Path] = None):
        """
        Initialize benchmark tracker.
        
        Args:
            baseline_file: Path to baseline metrics file
        """
        self.baseline_file = baseline_file or Path("tests/performance/baseline_metrics.json")
        self.results: List[BenchmarkResult] = []
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Load existing baselines if file exists
        if self.baseline_file.exists():
            self.load_baselines()
    
    def load_baselines(self):
        """Load baseline metrics from file."""
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
                for name, baseline_data in data.items():
                    self.baselines[name] = PerformanceBaseline(**baseline_data)
        except Exception as e:
            print(f"Warning: Could not load baselines: {e}")
    
    def save_baselines(self):
        """Save baseline metrics to file."""
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: asdict(baseline)
            for name, baseline in self.baselines.items()
        }
        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_result(self, result: BenchmarkResult):
        """Record a benchmark result."""
        self.results.append(result)
    
    def check_regression(self, test_name: str, elapsed_time: float, 
                        memory_usage: Optional[float] = None) -> bool:
        """
        Check if performance has regressed compared to baseline.
        
        Args:
            test_name: Name of the test
            elapsed_time: Elapsed time in seconds
            memory_usage: Memory usage in MB (optional)
            
        Returns:
            True if performance is acceptable, False if regressed
        """
        if test_name not in self.baselines:
            return True  # No baseline, assume OK
        
        baseline = self.baselines[test_name]
        
        # Check time regression (allow 20% overhead)
        if elapsed_time > baseline.max_elapsed_time * 1.2:
            return False
        
        # Check memory regression if provided
        if memory_usage and baseline.max_memory_usage:
            if memory_usage > baseline.max_memory_usage * 1.2:
                return False
        
        return True
    
    def update_baseline(self, test_name: str, max_elapsed_time: float,
                       max_memory_usage: Optional[float] = None):
        """Update baseline metrics for a test."""
        self.baselines[test_name] = PerformanceBaseline(
            test_name=test_name,
            max_elapsed_time=max_elapsed_time,
            max_memory_usage=max_memory_usage
        )
        self.save_baselines()
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        report_lines = [
            "# Performance Benchmark Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"Total benchmarks: {len(self.results)}",
            ""
        ]
        
        # Group by test name
        by_test: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)
        
        for test_name, results in by_test.items():
            report_lines.append(f"## {test_name}")
            avg_time = sum(r.elapsed_time for r in results) / len(results)
            min_time = min(r.elapsed_time for r in results)
            max_time = max(r.elapsed_time for r in results)
            
            report_lines.append(f"- Average time: {avg_time:.3f}s")
            report_lines.append(f"- Min time: {min_time:.3f}s")
            report_lines.append(f"- Max time: {max_time:.3f}s")
            
            if test_name in self.baselines:
                baseline = self.baselines[test_name]
                report_lines.append(f"- Baseline: {baseline.max_elapsed_time:.3f}s")
                if avg_time > baseline.max_elapsed_time * 1.2:
                    report_lines.append("  ⚠️  PERFORMANCE REGRESSION DETECTED")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, output_file: Path):
        """Save performance report to file."""
        report = self.generate_report()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)


# Global benchmark instance
_benchmark = None


def get_benchmark() -> PerformanceBenchmark:
    """Get or create global benchmark instance."""
    global _benchmark
    if _benchmark is None:
        _benchmark = PerformanceBenchmark()
    return _benchmark



