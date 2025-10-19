"""
Integration tests for the API layer.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_api.main import app
from pathwaylens_api.routes.analyze import router as analyze_router
from pathwaylens_api.routes.compare import router as compare_router
from pathwaylens_api.routes.visualize import router as visualize_router
from pathwaylens_api.routes.normalize import router as normalize_router


class TestAPIIntegration:
    """Integration tests for the API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_info_endpoint(self, client):
        """Test the info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "name" in data

    @pytest.mark.asyncio
    async def test_analyze_endpoint(self, client):
        """Test the analyze endpoint."""
        with patch('pathwaylens_api.tasks.analyze.analyze_task') as mock_task:
            mock_task.delay.return_value = Mock(id="test-job-id")
            
            response = client.post("/analyze", json={
                "input_data": ["GENE1", "GENE2", "GENE3"],
                "analysis_type": "ora",
                "database": "kegg",
                "species": "human"
            })
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test-job-id"

    @pytest.mark.asyncio
    async def test_compare_endpoint(self, client):
        """Test the compare endpoint."""
        with patch('pathwaylens_api.tasks.compare.compare_task') as mock_task:
            mock_task.delay.return_value = Mock(id="test-job-id")
            
            response = client.post("/compare", json={
                "input_data": ["file1.csv", "file2.csv"],
                "comparison_type": "overlap",
                "parameters": {}
            })
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test-job-id"

    @pytest.mark.asyncio
    async def test_visualize_endpoint(self, client):
        """Test the visualize endpoint."""
        with patch('pathwaylens_api.tasks.visualize.visualize_task') as mock_task:
            mock_task.delay.return_value = Mock(id="test-job-id")
            
            response = client.post("/visualize", json={
                "input_data": "analysis_result.json",
                "plot_type": "bar",
                "parameters": {}
            })
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test-job-id"

    @pytest.mark.asyncio
    async def test_normalize_endpoint(self, client):
        """Test the normalize endpoint."""
        with patch('pathwaylens_api.tasks.normalize.normalize_task') as mock_task:
            mock_task.delay.return_value = Mock(id="test-job-id")
            
            response = client.post("/normalize", json={
                "input_data": "gene_list.txt",
                "input_format": "gene_list",
                "output_format": "entrezgene",
                "species": "human"
            })
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test-job-id"

    def test_job_status_endpoint(self, client):
        """Test the job status endpoint."""
        with patch('pathwaylens_api.utils.database.get_job_status') as mock_get_status:
            mock_get_status.return_value = {
                "status": "completed",
                "result": {"pathways": []}
            }
            
            response = client.get("/jobs/test-job-id/status")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"

    def test_job_result_endpoint(self, client):
        """Test the job result endpoint."""
        with patch('pathwaylens_api.utils.database.get_job_result') as mock_get_result:
            mock_get_result.return_value = {"pathways": []}
            
            response = client.get("/jobs/test-job-id/result")
            assert response.status_code == 200
            data = response.json()
            assert "pathways" in data

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/analyze")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_rate_limiting(self, client):
        """Test rate limiting is applied."""
        # This would require more sophisticated testing with actual rate limiting
        # For now, just test that the endpoint responds
        response = client.get("/info")
        assert response.status_code == 200
