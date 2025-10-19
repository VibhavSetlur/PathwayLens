"""
API client for PathwayLens CLI.
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path


class APIClient:
    """Client for interacting with PathwayLens API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.logger = logger.bind(module="api_client")
    
    async def upload_file(
        self,
        file_path: str,
        file_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Upload a file to the API.
        
        Args:
            file_path: Path to the file to upload
            file_type: Type of file (auto, csv, excel, json)
            
        Returns:
            Upload response dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename=Path(file_path).name)
                    data.add_field('file_type', file_type)
                    
                    headers = {}
                    if self.api_key:
                        headers['Authorization'] = f'Bearer {self.api_key}'
                    
                    async with session.post(
                        f"{self.base_url}/upload",
                        data=data,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.logger.info(f"File uploaded successfully: {file_path}")
                            return result
                        else:
                            error_text = await response.text()
                            raise Exception(f"Upload failed: {error_text}")
                            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise
    
    async def normalize_data(
        self,
        file_id: str,
        input_format: str,
        output_format: str,
        species: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Normalize data using the API.
        
        Args:
            file_id: ID of the uploaded file
            input_format: Input format
            output_format: Output format
            species: Species for normalization
            **kwargs: Additional parameters
            
        Returns:
            Normalization response dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "file_id": file_id,
                    "input_format": input_format,
                    "output_format": output_format,
                    "species": species,
                    **kwargs
                }
                
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.post(
                    f"{self.base_url}/normalize",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Data normalization started: {file_id}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Normalization failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to normalize data: {e}")
            raise
    
    async def analyze_data(
        self,
        file_id: str,
        analysis_type: str,
        databases: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze data using the API.
        
        Args:
            file_id: ID of the uploaded file
            analysis_type: Type of analysis
            databases: List of databases to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis response dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "file_id": file_id,
                    "analysis_type": analysis_type,
                    "databases": databases,
                    **kwargs
                }
                
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.post(
                    f"{self.base_url}/analyze",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Data analysis started: {file_id}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Analysis failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to analyze data: {e}")
            raise
    
    async def compare_datasets(
        self,
        file_ids: List[str],
        comparison_type: str = "overlap",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare datasets using the API.
        
        Args:
            file_ids: List of file IDs to compare
            comparison_type: Type of comparison
            **kwargs: Additional parameters
            
        Returns:
            Comparison response dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "file_ids": file_ids,
                    "comparison_type": comparison_type,
                    **kwargs
                }
                
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.post(
                    f"{self.base_url}/compare",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Dataset comparison started: {file_ids}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Comparison failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to compare datasets: {e}")
            raise
    
    async def visualize_results(
        self,
        job_id: str,
        visualization_type: str = "dot_plot",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create visualizations using the API.
        
        Args:
            job_id: ID of the analysis job
            visualization_type: Type of visualization
            **kwargs: Additional parameters
            
        Returns:
            Visualization response dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "job_id": job_id,
                    "visualization_type": visualization_type,
                    **kwargs
                }
                
                headers = {'Content-Type': 'application/json'}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.post(
                    f"{self.base_url}/visualize",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Visualization started: {job_id}")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Visualization failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            raise
    
    async def get_job_status(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get job status from the API.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job status dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.get(
                    f"{self.base_url}/jobs/{job_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get job status: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")
            raise
    
    async def get_job_results(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get job results from the API.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job results dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.get(
                    f"{self.base_url}/results/{job_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get job results: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get job results: {e}")
            raise
    
    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List jobs from the API.
        
        Args:
            status: Filter by job status
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {"limit": limit}
                if status:
                    params["status"] = status
                
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.get(
                    f"{self.base_url}/jobs",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("jobs", [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to list jobs: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to list jobs: {e}")
            raise
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information from the API.
        
        Returns:
            System information dictionary
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                async with session.get(
                    f"{self.base_url}/info",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get system info: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            raise
    
    def set_api_key(self, api_key: str):
        """
        Set API key for authentication.
        
        Args:
            api_key: API key
        """
        self.api_key = api_key
        self.logger.info("API key set")
    
    def set_base_url(self, base_url: str):
        """
        Set base URL for the API.
        
        Args:
            base_url: Base URL
        """
        self.base_url = base_url.rstrip('/')
        self.logger.info(f"Base URL set to: {self.base_url}")
