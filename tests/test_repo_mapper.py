import unittest
import os
import tempfile
import shutil
from app.agents.repo_mapper import map_repo

class TestRepoMapper(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create dummy file structure
        os.makedirs(os.path.join(self.test_dir, "src"))
        with open(os.path.join(self.test_dir, "main.py"), "w") as f:
            f.write("if __name__ == '__main__':\n    print('hello')")
        with open(os.path.join(self.test_dir, "requirements.txt"), "w") as f:
            f.write("flask\nnumpy")
        with open(os.path.join(self.test_dir, "src", "utils.py"), "w") as f:
            f.write("def foo(): pass")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_map_repo(self):
        repo_map = map_repo(self.test_dir, "test_repo")
        
        self.assertEqual(repo_map.repo_name, "test_repo")
        self.assertIn("main.py", repo_map.entry_points)
        self.assertIn("flask", repo_map.dependencies)
        self.assertIn("src", repo_map.structure)
        self.assertEqual(repo_map.structure["src"]["utils.py"], "file")

if __name__ == '__main__':
    unittest.main()
