import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from app.agents.chat_agent import chat_with_agent, tools_map
from app.models import ChatRequest, Message

class TestChatAgent(unittest.IsolatedAsyncioTestCase):

    @patch('app.agents.chat_agent.genai.Client')
    @patch('app.agents.chat_agent.os.getenv')
    async def test_chat_with_agent(self, mock_getenv, mock_client_cls):
        # Setup mocks
        mock_getenv.return_value = "test-project"
        
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.candidates = [
            MagicMock(content=MagicMock(parts=[MagicMock(text="Hello!", function_call=None)]))
        ]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15
        )
        
        mock_client.models.generate_content.return_value = mock_response
        
        request = ChatRequest(messages=[Message(role="user", content="Hi")], model="test-model")
        
        response = await chat_with_agent(request)
        
        self.assertEqual(response.status, "success")
        self.assertEqual(response.response, "Hello!")
        self.assertEqual(response.usage.total_tokens, 15)

    def test_tools(self):
        # Test filesystem tools
        with patch('app.tools.filesystem.os.listdir') as mock_listdir:
            mock_listdir.return_value = ["file1.py", "dir1"]
            with patch('app.tools.filesystem.os.path.isdir') as mock_isdir:
                mock_isdir.side_effect = [False, True] # file1 is file, dir1 is dir
                with patch('app.tools.filesystem.os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    result = tools_map["list_files"](".")
                    self.assertIn("[FILE] file1.py", result)
                    self.assertIn("[DIR] dir1", result)

if __name__ == '__main__':
    unittest.main()
