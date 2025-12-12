import requests
import json
import sys

def test_chat_stream():
    url = "http://localhost:8001/chat-stream"
    
    # Test message
    payload = {
        "messages": [
            {"role": "user", "content": "What files are in the current directory?"}
        ],
        "model": "gemini-3-pro-preview"
    }
    
    print(f"Connecting to {url}...")
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return

            print("Connected. Streaming response:")
            print("-" * 50)
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    try:
                        data = json.loads(line_text)
                        print(f"Event: {data.get('type')}")
                        if data.get('type') == 'token':
                            print(f"  Content: {repr(data.get('content'))}")
                        elif data.get('type') == 'tool_start':
                            print(f"  Tool: {data.get('tool')}")
                            print(f"  Args: {data.get('args')}")
                        elif data.get('type') == 'tool_end':
                            print(f"  Tool: {data.get('tool')}")
                            result = data.get('result', '')
                            print(f"  Result: {result[:100] if len(result) > 100 else result}...")
                        elif data.get('type') == 'usage':
                            print(f"  Usage: {data.get('usage')}")
                        elif data.get('type') == 'error':
                            print(f"  Error: {data.get('content')}")
                    except json.JSONDecodeError:
                        print(f"Failed to parse: {line_text}")
            
            print("-" * 50)
            print("Stream finished.")

    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_chat_stream()
