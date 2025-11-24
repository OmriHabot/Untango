import unittest
import os
import json
import shutil
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock
from app.ingest_manager import IngestManager

class TestIngestManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cache_file = ".ingest_cache.json"
        
        # Create a dummy file
        with open(os.path.join(self.test_dir, "test.py"), "w") as f:
            f.write("print('hello')")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)

    @patch('app.ingest_manager.chunk_python_code')
    @patch('app.ingest_manager.get_collection')
    @patch('app.ingest_manager.delete_file_chunks')
    async def test_sync_repo(self, mock_delete, mock_get_collection, mock_chunk):
        # Setup mocks
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_chunk.return_value = [{"id": "1", "content": "code", "metadata": {"key": "val"}}]
        
        # Initialize manager
        manager = IngestManager(repo_path=self.test_dir)
        
        # 1. First sync (should ingest)
        await manager.sync_repo()
        
        mock_delete.assert_called()
        mock_chunk.assert_called()
        mock_collection.add.assert_called()
        
        # Verify cache was created
        self.assertTrue(os.path.exists(self.cache_file))
        with open(self.cache_file, 'r') as f:
            cache = json.load(f)
            self.assertIn("test.py", cache)
            
        # 2. Second sync (no changes, should NOT ingest)
        mock_delete.reset_mock()
        mock_collection.add.reset_mock()
        
        await manager.sync_repo()
        
        mock_delete.assert_not_called()
        mock_collection.add.assert_not_called()
        
        # 3. Modify file (should ingest)
        # Wait a bit to ensure mtime changes (filesystems can be coarse)
        import time
        time.sleep(0.01)
        os.utime(os.path.join(self.test_dir, "test.py"), None)
        
        await manager.sync_repo()
        
        mock_delete.assert_called()
        mock_collection.add.assert_called()

if __name__ == '__main__':
    unittest.main()
