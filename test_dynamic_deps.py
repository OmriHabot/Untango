import os
import shutil
import unittest
from app.agents.repo_mapper import map_repo

class TestDynamicDeps(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_repo_dynamic"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dynamic_detection(self):
        # Create a python file with imports
        with open(os.path.join(self.test_dir, "main.py"), "w") as f:
            f.write("import pandas\nfrom flask import Flask\nimport os\n")
            
        # Ensure no requirements.txt
        req_path = os.path.join(self.test_dir, "requirements.txt")
        if os.path.exists(req_path):
            os.remove(req_path)
            
        # Run mapper
        repo_map = map_repo(self.test_dir, "TestRepo")
        
        print(f"Detected Dependencies: {repo_map.dependencies}")
        
        # Verify
        self.assertIn("pandas", repo_map.dependencies)
        self.assertIn("flask", repo_map.dependencies)
        self.assertNotIn("os", repo_map.dependencies) # Should be filtered as stdlib

if __name__ == '__main__':
    unittest.main()
