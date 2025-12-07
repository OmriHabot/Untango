import unittest
from unittest.mock import MagicMock, patch
import asyncio
from app.models import EnvInfo, RepoMap
from app.context_manager import DependencyStatus
from app.agents.runbook_generator import generate_runbook_content

class TestRunbookPrompt(unittest.TestCase):
    @patch('app.agents.runbook_generator.genai.Client')
    @patch('app.agents.runbook_generator.perform_hybrid_search')
    def test_prompt_content(self, mock_search, mock_client_cls):
        # 1. Setup Mock Data
        repo_map = RepoMap(
            repo_name="LegacyApp", root_path=".", structure={"src": ["main.py"]}, 
            entry_points=["src/main.py"], dependencies=["tensorflow==1.15.0"],
            last_updated="2019-05-20" # VERY OLD
        )
        env_info = EnvInfo(
            os_info="Mac", python_version="3.11", cuda_available=False, gpu_info="",
            installed_packages=["tensorflow==2.14.0"]
        )
        deps = [
            DependencyStatus("tensorflow", "==1.15.0", "2.14.0", "MISMATCH")
        ]
        
        # 2. Mock RAG Results
        mock_search.return_value = [
            {"metadata": {"filepath": "README.md"}, "content": "Run with: python src/main.py"}
        ]
        
        # 3. Mock LLM Client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Mock Runbook Content"
        mock_client.models.generate_content.return_value = mock_response

        # 4. Run Generator
        asyncio.run(generate_runbook_content(
            repo_map, env_info, deps, "test-project", "global"
        ))

        # 5. Verify Prompt Content
        # Get the call args
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args.kwargs['contents']
        
        print("\n=== GENERATED PROMPT ===")
        print(prompt_sent)
        print("========================\n")

        # Assertions for Quality
        self.assertIn("Last Updated: 2019-05-20", prompt_sent)
        self.assertIn("tensorflow: Status=MISMATCH", prompt_sent)
        self.assertIn("Run with: python src/main.py", prompt_sent) # RAG content
        self.assertIn("Time Rot Warning", prompt_sent) # The instruction to warn
        self.assertIn("Dependency Analysis (CRITICAL)", prompt_sent)

if __name__ == '__main__':
    unittest.main()
