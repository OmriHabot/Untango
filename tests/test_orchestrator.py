import unittest
from unittest.mock import patch, AsyncMock
from app.orchestrator import generate_runbook_orchestrator
from app.models import RunbookRequest, EnvInfo, RepoMap

class TestOrchestrator(unittest.IsolatedAsyncioTestCase):

    @patch('app.orchestrator.scan_environment')
    @patch('app.orchestrator.map_repo')
    @patch('app.orchestrator.generate_runbook_content')
    @patch('app.orchestrator.os.path.exists')
    @patch('app.orchestrator.os.getenv')
    async def test_generate_runbook_orchestrator(self, mock_getenv, mock_exists, mock_gen_content, mock_map, mock_scan):
        # Setup mocks
        mock_exists.return_value = True
        mock_getenv.return_value = "test-project" # for GOOGLE_CLOUD_PROJECT
        
        mock_scan.return_value = EnvInfo(
            os_info="Linux", python_version="3.9", cuda_available=False, gpu_info="", installed_packages=[]
        )
        mock_map.return_value = RepoMap(
            repo_name="test", root_path="/tmp", structure={}, entry_points=[], dependencies=[]
        )
        mock_gen_content.return_value = "# Runbook"
        
        request = RunbookRequest(repo_path="/tmp/test", repo_name="test")
        
        response = await generate_runbook_orchestrator(request)
        
        self.assertEqual(response.status, "success")
        self.assertEqual(response.runbook, "# Runbook")
        
        mock_scan.assert_called_once()
        mock_map.assert_called_once()
        mock_gen_content.assert_called_once()

if __name__ == '__main__':
    unittest.main()
