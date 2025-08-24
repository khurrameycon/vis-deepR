# examples/api_client_example.py
"""
Example API client showing how external applications can interact with the multi-user Browser Use service
"""

import asyncio
import aiohttp
import websockets
import json
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrowserUseAPIClient:
    """Client for interacting with the Browser Use API"""
    
    def __init__(self, base_url: str = "http://localhost:7788"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session_id:
            await self.delete_session()
        if self.session:
            await self.session.close()
    
    async def create_session(self) -> str:
        """Create a new session and return session ID"""
        async with self.session.post(f"{self.base_url}/api/session/create") as response:
            if response.status == 200:
                data = await response.json()
                self.session_id = data["session_id"]
                logger.info(f"Created session: {self.session_id}")
                return self.session_id
            else:
                raise Exception(f"Failed to create session: {response.status}")
    
    async def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        if not self.session_id:
            raise Exception("No active session")
        
        async with self.session.get(f"{self.base_url}/api/session/{self.session_id}") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get session info: {response.status}")
    
    async def delete_session(self) -> bool:
        """Delete the current session"""
        if not self.session_id:
            return False
        
        async with self.session.delete(f"{self.base_url}/api/session/{self.session_id}") as response:
            if response.status == 200:
                logger.info(f"Deleted session: {self.session_id}")
                self.session_id = None
                return True
            else:
                logger.error(f"Failed to delete session: {response.status}")
                return False
    
    async def run_agent(self, 
                       task: str,
                       llm_provider: str = "google",
                       llm_model_name: Optional[str] = None,
                       llm_temperature: float = 0.6,
                       llm_base_url: Optional[str] = None,
                       llm_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Start an agent task"""
        if not self.session_id:
            raise Exception("No active session")
        
        payload = {
            "session_id": self.session_id,
            "task": task,
            "llm_provider": llm_provider,
            "llm_temperature": llm_temperature
        }
        
        if llm_model_name:
            payload["llm_model_name"] = llm_model_name
        if llm_base_url:
            payload["llm_base_url"] = llm_base_url
        if llm_api_key:
            payload["llm_api_key"] = llm_api_key
        
        async with self.session.post(f"{self.base_url}/api/agent/run", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Agent started: {data}")
                return data
            else:
                error_text = await response.text()
                raise Exception(f"Failed to start agent: {response.status} - {error_text}")
    
    async def stop_agent(self) -> Dict[str, Any]:
        """Stop the currently running agent"""
        if not self.session_id:
            raise Exception("No active session")
        
        async with self.session.post(f"{self.base_url}/api/agent/stop/{self.session_id}") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Agent stopped: {data}")
                return data
            else:
                raise Exception(f"Failed to stop agent: {response.status}")
    
    async def stream_data(self, message_handler=None):
        """Connect to WebSocket and stream data"""
        if not self.session_id:
            raise Exception("No active session")
        
        ws_url = f"ws://localhost:7788/ws/stream/{self.session_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Connected to WebSocket for session {self.session_id}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if message_handler:
                            await message_handler(data)
                        else:
                            # Default message handling
                            msg_type = data.get("type", "unknown")
                            msg_data = data.get("data", "")
                            
                            if msg_type == "log":
                                logger.info(f"[AGENT LOG] {msg_data}")
                            elif msg_type == "result":
                                logger.info(f"[RESULT] {msg_data}")
                                break  # Task completed
                            elif msg_type == "error":
                                logger.error(f"[ERROR] {msg_data}")
                                break
                            elif msg_type == "stream":
                                logger.debug("Received screenshot data")
                            else:
                                logger.info(f"[{msg_type.upper()}] {msg_data}")
                    
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")

# Example usage functions
async def simple_example():
    """Simple example of using the API client"""
    async with BrowserUseAPIClient() as client:
        # Create session
        session_id = await client.create_session()
        print(f"Created session: {session_id}")
        
        # Get session info
        info = await client.get_session_info()
        print(f"Session info: {info}")
        
        # Start agent task
        task_result = await client.run_agent(
            task="Go to google.com and search for 'Python asyncio tutorial'",
            llm_provider="google"
        )
        print(f"Agent started: {task_result}")
        
        # Stream the results
        await client.stream_data()

async def advanced_example():
    """Advanced example with custom message handling"""
    
    async def custom_message_handler(data):
        """Custom handler for WebSocket messages"""
        msg_type = data.get("type")
        msg_data = data.get("data")
        
        if msg_type == "log":
            print(f"🤖 Agent: {msg_data}")
        elif msg_type == "result":
            print(f"✅ Final Result: {msg_data}")
        elif msg_type == "error":
            print(f"❌ Error: {msg_data}")
        elif msg_type == "stream":
            print("📸 Screenshot received")
    
    async with BrowserUseAPIClient() as client:
        # Create session
        await client.create_session()
        
        # Start multiple tasks (one at a time per session)
        tasks = [
            "Go to wikipedia.org and search for 'Artificial Intelligence'",
            "Go to github.com and search for 'python web scraping'",
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n--- Running Task {i} ---")
            await client.run_agent(task=task)
            await client.stream_data(custom_message_handler)
            
            if i < len(tasks):
                print("Waiting before next task...")
                await asyncio.sleep(2)

# Curl command examples (for reference)
def print_curl_examples():
    """Print equivalent curl commands"""
    print("""
    # Equivalent curl commands:
    
    # 1. Create session
    curl -X POST http://localhost:7788/api/session/create
    
    # 2. Get session info (replace SESSION_ID)
    curl -X GET http://localhost:7788/api/session/SESSION_ID
    
    # 3. Start agent task with different LLM providers
    curl -X POST http://localhost:7788/api/agent/run \\
      -H "Content-Type: application/json" \\
      -d '{
        "session_id": "SESSION_ID",
        "task": "Go to google.com and search for AI news",
        "llm_provider": "openai",
        "llm_model_name": "gpt-4o",
        "llm_api_key": "sk-your-openai-key-here",
        "llm_temperature": 0.7
      }'
    
    # Example with Google/Gemini
    curl -X POST http://localhost:7788/api/agent/run \\
      -H "Content-Type: application/json" \\
      -d '{
        "session_id": "SESSION_ID",
        "task": "Navigate to wikipedia and search for Python programming",
        "llm_provider": "google",
        "llm_model_name": "gemini-2.0-flash",
        "llm_api_key": "your-google-api-key-here"
      }'
    
    # Example with Anthropic Claude
    curl -X POST http://localhost:7788/api/agent/run \\
      -H "Content-Type: application/json" \\
      -d '{
        "session_id": "SESSION_ID",
        "task": "Go to github.com and search for browser automation",
        "llm_provider": "anthropic",
        "llm_model_name": "claude-3-5-sonnet-20241022",
        "llm_api_key": "sk-ant-your-anthropic-key-here"
      }'
    
    # Example with DeepSeek
    curl -X POST http://localhost:7788/api/agent/run \\
      -H "Content-Type: application/json" \\
      -d '{
        "session_id": "SESSION_ID",
        "task": "Search for latest AI research papers",
        "llm_provider": "deepseek",
        "llm_model_name": "deepseek-chat",
        "llm_api_key": "your-deepseek-key-here",
        "llm_base_url": "https://api.deepseek.com"
      }'
    
    # 4. Stop agent
    curl -X POST http://localhost:7788/api/agent/stop/SESSION_ID
    
    # 5. Delete session
    curl -X DELETE http://localhost:7788/api/session/SESSION_ID
    
    # 6. Health check
    curl -X GET http://localhost:7788/health
    """)

if __name__ == "__main__":
    print("Browser Use API Client Examples")
    print("=" * 40)
    
    # Print curl examples
    print_curl_examples()
    
    # Run simple example
    print("\nRunning simple example...")
    asyncio.run(simple_example())