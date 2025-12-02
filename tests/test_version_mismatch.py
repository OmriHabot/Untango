import unittest
from app.context_manager import ContextManager, DependencyStatus
from app.models import EnvInfo, RepoMap

class TestVersionMismatch(unittest.TestCase):
    def test_detect_mismatch(self):
        # 1. Mock Environment (System has NEW version)
        env = EnvInfo(
            os_info="Mac", python_version="3.9", cuda_available=False, gpu_info="",
            installed_packages=["transformers==4.30.0", "torch==2.0.0"]
        )
        
        # 2. Mock Repository (Repo needs OLD version)
        # Case A: Explicit requirement in requirements.txt
        repo = RepoMap(
            repo_name="OldRepo", root_path=".", structure={}, entry_points=[],
            dependencies=["transformers==2.5.0"], # Explicit old version
            last_updated="2020-01-01"
        )
        
        # 3. Analyze
        cm = ContextManager()
        analysis = cm.analyze_dependencies(env, repo)
        
        # 4. Verify Mismatch Detection
        transformers_status = next((d for d in analysis if d.package == "transformers"), None)
        
        self.assertIsNotNone(transformers_status)
        self.assertEqual(transformers_status.status, "MISMATCH")
        self.assertEqual(transformers_status.installed_version, "4.30.0")
        self.assertEqual(transformers_status.required_version, "==2.5.0")
        
        print("\nTest Result:")
        print(f"Package: {transformers_status.package}")
        print(f"Required: {transformers_status.required_version}")
        print(f"Installed: {transformers_status.installed_version}")
        print(f"Status: {transformers_status.status}")

if __name__ == '__main__':
    unittest.main()
