import unittest
from unittest.mock import patch, MagicMock
from app.agents.env_scanner import scan_environment, get_os_info, get_python_version

class TestEnvScanner(unittest.TestCase):

    def test_get_os_info(self):
        info = get_os_info()
        self.assertIsInstance(info, str)
        self.assertTrue(len(info) > 0)

    def test_get_python_version(self):
        version = get_python_version()
        self.assertIsInstance(version, str)
        self.assertTrue("." in version)

    @patch('app.agents.env_scanner.subprocess.run')
    def test_scan_environment_mocked(self, mock_run):
        # Mock nvidia-smi failure
        mock_run.side_effect = [
            MagicMock(returncode=1), # nvidia-smi
            MagicMock(returncode=0, stdout="package==1.0.0\n") # pip freeze
        ]
        
        env_info = scan_environment()
        
        self.assertFalse(env_info.cuda_available)
        self.assertEqual(env_info.installed_packages, ["package==1.0.0"])
        self.assertTrue(len(env_info.os_info) > 0)

if __name__ == '__main__':
    unittest.main()
