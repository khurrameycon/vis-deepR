# practical_integration_example.py
"""
Practical example showing how a real application would integrate with the Multi-User Browser Use API
This demonstrates a complete workflow with different LLM providers and real error handling.
"""

import asyncio
import aiohttp
import websockets
import json
import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    OLLAMA = "ollama"

@dataclass
class LLMConfig:
    """Configuration for LLM"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.6

class BrowserAutomationService:
    """Production-ready service for browser automation with multiple LLM providers"""
    
    def __init__(self, api_base_url: str = "http://localhost:7788"):
        self.api_base_url = api_base_url
        self.session = None
        
        # Validate API key if required
        if llm_provider != LLMProvider.OLLAMA and not llm_config.api_key:
            raise ValueError(f"API key required for {llm_provider.value} but not provided")
        
        # Prepare request payload
        payload = {
            "session_id": session_id,
            "task": task,
            "llm_provider": llm_config.provider.value,
            "llm_model_name": llm_config.model_name,
            "llm_temperature": llm_config.temperature
        }
        
        # Add optional fields
        if llm_config.api_key:
            payload["llm_api_key"] = llm_config.api_key
        if llm_config.base_url:
            payload["llm_base_url"] = llm_config.base_url
        
        # Start the automation task
        async with self.session.post(f"{self.api_base_url}/api/agent/run", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Automation task started: {data}")
                return data
            else:
                error_text = await response.text()
                raise Exception(f"Failed to start automation: {response.status} - {error_text}")
    
    async def monitor_task_progress(self, session_id: str, callback=None) -> Optional[str]:
        """Monitor task progress via WebSocket and return final result"""
        ws_url = f"ws://localhost:7788/ws/stream/{session_id}"
        final_result = None
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"Monitoring task progress for session {session_id}")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        msg_data = data.get("data")
                        
                        if callback:
                            await callback(msg_type, msg_data)
                        
                        if msg_type == "result":
                            final_result = msg_data
                            logger.info(f"Task completed with result: {final_result}")
                            break
                        elif msg_type == "error":
                            logger.error(f"Task failed with error: {msg_data}")
                            raise Exception(f"Task execution error: {msg_data}")
                        elif msg_type == "log":
                            logger.debug(f"Agent log: {msg_data}")
                    
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON received: {message}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        break
        
        except Exception as e:
            logger.error(f"WebSocket monitoring failed: {e}")
            raise
        
        return final_result
    
    async def stop_task(self, session_id: str) -> bool:
        """Stop a running automation task"""
        async with self.session.post(f"{self.api_base_url}/api/agent/stop/{session_id}") as response:
            if response.status == 200:
                logger.info(f"Task stopped for session {session_id}")
                return True
            else:
                logger.error(f"Failed to stop task: {response.status}")
                return False
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information about a session"""
        async with self.session.get(f"{self.api_base_url}/api/session/{session_id}") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get session info: {response.status}")
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a browser session"""
        async with self.session.delete(f"{self.api_base_url}/api/session/{session_id}") as response:
            if response.status == 200:
                logger.info(f"Session cleaned up: {session_id}")
                return True
            else:
                logger.error(f"Failed to cleanup session: {response.status}")
                return False
    
    async def run_complete_workflow(self,
                                   task: str,
                                   llm_provider: LLMProvider,
                                   progress_callback=None) -> Optional[str]:
        """Run a complete automation workflow from start to finish"""
        session_id = None
        try:
            # Create session
            session_id = await self.create_browser_session()
            
            # Start task
            await self.run_automation_task(session_id, task, llm_provider)
            
            # Monitor progress and get result
            result = await self.monitor_task_progress(session_id, progress_callback)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            if session_id:
                await self.stop_task(session_id)
            raise
        finally:
            # Always cleanup
            if session_id:
                await self.cleanup_session(session_id)

# Example usage functions
async def example_basic_automation():
    """Basic automation example"""
    print("🤖 Basic Automation Example")
    print("=" * 40)
    
    async def progress_handler(msg_type, msg_data):
        if msg_type == "log":
            print(f"[AGENT] {msg_data}")
        elif msg_type == "stream":
            print("[STREAM] Screenshot received")
    
    async with BrowserAutomationService() as service:
        # Check service health
        health = await service.check_service_health()
        print(f"Service status: {health}")
        
        # Run automation with different providers
        tasks = [
            ("Search for Python tutorials", LLMProvider.GOOGLE),
            ("Find the latest AI news", LLMProvider.OPENAI),
            ("Research quantum computing", LLMProvider.ANTHROPIC),
        ]
        
        for task, provider in tasks:
            try:
                print(f"\n🚀 Running: {task} with {provider.value}")
                result = await service.run_complete_workflow(
                    task=task,
                    llm_provider=provider,
                    progress_callback=progress_handler
                )
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Failed: {e}")

async def example_parallel_automation():
    """Example of running multiple automations in parallel"""
    print("\n🔄 Parallel Automation Example")
    print("=" * 40)
    
    async def run_single_task(service, task, provider):
        try:
            session_id = await service.create_browser_session()
            await service.run_automation_task(session_id, task, provider)
            result = await service.monitor_task_progress(session_id)
            await service.cleanup_session(session_id)
            return f"{provider.value}: {result}"
        except Exception as e:
            return f"{provider.value}: Failed - {e}"
    
    async with BrowserAutomationService() as service:
        # Run multiple tasks in parallel
        tasks = [
            ("Search for machine learning courses", LLMProvider.GOOGLE),
            ("Find Python job openings", LLMProvider.OPENAI),
            ("Research React frameworks", LLMProvider.ANTHROPIC),
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[
            run_single_task(service, task, provider)
            for task, provider in tasks
        ], return_exceptions=True)
        
        print("Results:")
        for result in results:
            print(f"  - {result}")

async def example_custom_llm_config():
    """Example with custom LLM configuration"""
    print("\n⚙️ Custom LLM Configuration Example")
    print("=" * 40)
    
    # Custom configuration for specific use case
    custom_config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY", "your-key-here"),
        base_url="https://api.deepseek.com",
        temperature=0.8  # Higher creativity
    )
    
    async with BrowserAutomationService() as service:
        session_id = await service.create_browser_session()
        
        try:
            await service.run_automation_task(
                session_id=session_id,
                task="Analyze the top 5 programming languages and their use cases",
                llm_provider=LLMProvider.DEEPSEEK,
                custom_llm_config=custom_config
            )
            
            result = await service.monitor_task_progress(session_id)
            print(f"Custom config result: {result}")
            
        finally:
            await service.cleanup_session(session_id)

def print_integration_guide():
    """Print integration guide for developers"""
    print("""
🔧 INTEGRATION GUIDE
==================

1. Set Environment Variables:
   export OPENAI_API_KEY="sk-your-openai-key"
   export GOOGLE_API_KEY="your-google-key"
   export ANTHROPIC_API_KEY="sk-ant-your-key"
   export DEEPSEEK_API_KEY="your-deepseek-key"
   export GROQ_API_KEY="your-groq-key"

2. Basic Integration Pattern:
   async with BrowserAutomationService() as service:
       session_id = await service.create_browser_session()
       await service.run_automation_task(session_id, task, llm_provider)
       result = await service.monitor_task_progress(session_id)
       await service.cleanup_session(session_id)

3. Error Handling:
   - Always use try/finally blocks
   - Clean up sessions even on errors
   - Handle WebSocket disconnections
   - Validate API keys before use

4. Production Considerations:
   - Implement proper logging
   - Add retry logic for transient failures
   - Monitor session limits
   - Use connection pooling for high throughput

5. Curl Equivalent:
   # Create session
   SESSION_ID=$(curl -s -X POST http://localhost:7788/api/session/create | jq -r '.session_id')
   
   # Start automation
   curl -X POST http://localhost:7788/api/agent/run \\
     -H "Content-Type: application/json" \\
     -d '{"session_id":"'$SESSION_ID'","task":"Your task","llm_provider":"google","llm_api_key":"your-key"}'
   
   # Monitor via WebSocket
   wscat -c ws://localhost:7788/ws/stream/$SESSION_ID
   
   # Cleanup
   curl -X DELETE http://localhost:7788/api/session/$SESSION_ID
""")

async def main():
    """Main demonstration function"""
    print("🌐 Multi-User Browser Automation Service")
    print("=" * 50)
    
    # Print integration guide
    print_integration_guide()
    
    # Run examples (uncomment the ones you want to test)
    
    # Basic example
    try:
        await example_basic_automation()
    except Exception as e:
        print(f"Basic automation failed: {e}")
    
    # Parallel example (uncomment to test)
    # try:
    #     await example_parallel_automation()
    # except Exception as e:
    #     print(f"Parallel automation failed: {e}")
    
    # Custom config example (uncomment to test)
    # try:
    #     await example_custom_llm_config()
    # except Exception as e:
    #     print(f"Custom config automation failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Browser Automation Service Examples")
    print("Make sure the API server is running: python webui.py")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples stopped by user")
    except Exception as e:
        print(f"\n💥 Examples failed: {e}")
        import traceback
        traceback.print_exc()
#  Predefined LLM configurations
        self.llm_configs = {
            LLMProvider.OPENAI: LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai.com/v1"
            ),
            LLMProvider.GOOGLE: LLMConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-2.0-flash",
                api_key=os.getenv("GOOGLE_API_KEY")
            ),
            LLMProvider.ANTHROPIC: LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url="https://api.anthropic.com"
            ),
            LLMProvider.DEEPSEEK: LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name="deepseek-reasoner",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            ),
            LLMProvider.GROQ: LLMConfig(
                provider=LLMProvider.GROQ,
                model_name="llama-3.1-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            ),
            LLMProvider.OLLAMA: LLMConfig(
                provider=LLMProvider.OLLAMA,
                model_name="qwen2.5:14b",
                base_url="http://localhost:11434"
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_service_health(self) -> Dict[str, Any]:
        """Check if the service is healthy and available"""
        try:
            async with self.session.get(f"{self.api_base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Service unhealthy: {response.status}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def get_available_providers(self) -> Dict[str, str]:
        """Get list of available LLM providers"""
        async with self.session.get(f"{self.api_base_url}/api/providers") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to get providers: {response.status}")
    
    async def create_browser_session(self) -> str:
        """Create a new browser session"""
        async with self.session.post(f"{self.api_base_url}/api/session/create") as response:
            if response.status == 200:
                data = await response.json()
                session_id = data["session_id"]
                logger.info(f"Created browser session: {session_id}")
                return session_id
            else:
                error_text = await response.text()
                raise Exception(f"Failed to create session: {response.status} - {error_text}")
    
    async def run_automation_task(self, 
                                 session_id: str,
                                 task: str,
                                 llm_provider: LLMProvider,
                                 custom_llm_config: Optional[LLMConfig] = None) -> Dict[str, Any]:
        """Run an automation task with specified LLM provider"""
        
        # Use custom config or default
        llm_config = custom_llm_config or self.llm_configs.get(llm_provider)
        if not llm_config:
            raise ValueError(f"No configuration found for provider: {llm_provider}")
        
        #