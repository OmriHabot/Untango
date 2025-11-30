import unittest
from unittest.mock import MagicMock, patch
from app.context_manager import ContextManager, ContextReport
from app.models import EnvInfo, RepoMap
from app.agents.repo_mapper import get_last_updated_date
import os
import time

class TestTimeInference(unittest.TestCase):
    def test_context_report_includes_date(self):
        env = EnvInfo(
            os_info="Mac", python_version="3.9", cuda_available=False, gpu_info="",
            installed_packages=[]
        )
        repo = RepoMap(
            repo_name="OldRepo", root_path=".", structure={}, entry_points=[],
            dependencies=[],
            last_updated="2020-01-01"
        )
        
        report = ContextReport(env, repo, [])
        report_str = report.to_string()
        
        self.assertIn("Last Updated: 2020-01-01", report_str)

    def test_get_last_updated_date_filesystem(self):
        # Create a temp file
        test_file = "temp_test_date.txt"
        with open(test_file, "w") as f:
            f.write("test")
            
        # Get date
        date = get_last_updated_date(".")
        
        # Verify it returns a date string (YYYY-MM-DD)
        self.assertRegex(date, r"\d{4}-\d{2}-\d{2}")
        
        os.remove(test_file)

if __name__ == '__main__':
    unittest.main()
