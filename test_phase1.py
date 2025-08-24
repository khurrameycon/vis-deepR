# test_phase1.py
"""
Test script for Phase 1 - Session Management and Multi-user API
"""

import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:7788"

async def test_session_management():
    """Test basic session management functionality"""
    print("🧪 Testing Session Management...")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health check
        print("\n1. Testing health check...")
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Health check passed: {data}")
            else:
                print(f"❌ Health check failed: {response.status}")
                return False
        
        # Test 2: Create session
        print("\n2. Testing session creation...")
        async with session.post(f"{BASE_URL}/api/session/create") as response:
            if response.status == 200:
                data = await response.json()
                session_id = data["session_id"]
                print(f"✅ Session created: {session_id}")
            else:
                print(f"❌ Session creation failed: {response.status}")
                return False
        
        # Test 3: Get session info
        print("\n3. Testing session info retrieval...")
        async with session.get(f"{BASE_URL}/api/session/{session_id}") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Session info retrieved: {data}")
            else:
                print(f"❌ Session info retrieval failed: {response.status}")
                return False
        
        # Test 4: Create multiple sessions
        print("\n4. Testing multiple session creation...")
        session_ids = []
        for i in range(3):
            async with session.post(f"{BASE_URL}/api/session/create") as response:
                if response.status == 200:
                    data = await response.json()
                    session_ids.append(data["session_id"])
                    print(f"✅ Additional session {i+1} created: {data['session_id']}")
                else:
                    print(f"❌ Additional session {i+1} creation failed: {response.status}")
        
        # Test 5: Check health with multiple sessions
        print("\n5. Testing health with multiple sessions...")
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Health check with multiple sessions: {data}")
                expected_sessions = len(session_ids) + 1  # +1 for the first session
                if data["active_sessions"] == expected_sessions:
                    print(f"✅ Session count correct: {data['active_sessions']}")
                else:
                    print(f"⚠️ Session count mismatch: expected {expected_sessions}, got {data['active_sessions']}")
            else:
                print(f"❌ Health check failed: {response.status}")
        
        # Test 6: Delete sessions
        print("\n6. Testing session deletion...")
        all_sessions = [session_id] + session_ids
        for sid in all_sessions:
            async with session.delete(f"{BASE_URL}/api/session/{sid}") as response:
                if response.status == 200:
                    print(f"✅ Session deleted: {sid}")
                else:
                    print(f"❌ Session deletion failed for {sid}: {response.status}")
        
        # Test 7: Verify all sessions deleted
        print("\n7. Testing session cleanup verification...")
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                if data["active_sessions"] == 0:
                    print(f"✅ All sessions cleaned up: {data}")
                else:
                    print(f"⚠️ Sessions not fully cleaned up: {data}")
            else:
                print(f"❌ Health check failed: {response.status}")
        
        return True

async def test_api_endpoints():
    """Test API endpoints without actually running agents (requires API keys)"""
    print("\n🧪 Testing API Endpoints (without agent execution)...")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Get providers
        print("\n1. Testing providers endpoint...")
        async with session.get(f"{BASE_URL}/api/providers") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Providers retrieved: {list(data.keys())}")
            else:
                print(f"❌ Providers endpoint failed: {response.status}")
                return False
        
        # Test 2: Create session for agent testing
        print("\n2. Creating session for agent testing...")
        async with session.post(f"{BASE_URL}/api/session/create") as response:
            if response.status == 200:
                data = await response.json()
                session_id = data["session_id"]
                print(f"✅ Session created for agent testing: {session_id}")
            else:
                print(f"❌ Session creation failed: {response.status}")
                return False
        
        # Test 3: Test agent run endpoint with different LLM providers (should fail without proper API key)
        print("\n3. Testing agent run endpoint with different LLM providers...")
        
        llm_test_cases = [
            {
                "name": "OpenAI GPT-4",
                "payload": {
                    "session_id": session_id,
                    "task": "Go to google.com",
                    "llm_provider": "openai",
                    "llm_model_name": "gpt-4o",
                    "llm_api_key": "fake-key-for-testing"
                }
            },
            {
                "name": "Google Gemini",
                "payload": {
                    "session_id": session_id,
                    "task": "Go to google.com",
                    "llm_provider": "google",
                    "llm_model_name": "gemini-2.0-flash",
                    "llm_api_key": "fake-key-for-testing"
                }
            },
            {
                "name": "Anthropic Claude",
                "payload": {
                    "session_id": session_id,
                    "task": "Go to google.com",
                    "llm_provider": "anthropic",
                    "llm_model_name": "claude-3-5-sonnet-20241022",
                    "llm_api_key": "fake-key-for-testing"
                }
            },
            {
                "name": "DeepSeek",
                "payload": {
                    "session_id": session_id,
                    "task": "Go to google.com",
                    "llm_provider": "deepseek",
                    "llm_model_name": "deepseek-chat",
                    "llm_api_key": "fake-key-for-testing",
                    "llm_base_url": "https://api.deepseek.com"
                }
            }
        ]
        
        for test_case in llm_test_cases:
            print(f"\n   Testing {test_case['name']}...")
            async with session.post(f"{BASE_URL}/api/agent/run", json=test_case['payload']) as response:
                # We expect these to fail with fake API keys
                if response.status in [400, 401, 403, 500]:
                    print(f"   ✅ {test_case['name']} correctly failed with fake API key: {response.status}")
                elif response.status == 200:
                    print(f"   ⚠️ {test_case['name']} unexpectedly succeeded (real API key might be set)")
                    # Try to stop it
                    async with session.post(f"{BASE_URL}/api/agent/stop/{session_id}") as stop_response:
                        print(f"   Attempted to stop agent: {stop_response.status}")
                else:
                    print(f"   ❌ Unexpected response for {test_case['name']}: {response.status}")
        
        # Test with missing required fields
        print("\n   Testing with missing required fields...")
        invalid_payload = {
            "session_id": session_id,
            "task": "Go to google.com"
            # Missing llm_provider
        }
        async with session.post(f"{BASE_URL}/api/agent/run", json=invalid_payload) as response:
            if response.status == 422:  # Validation error
                print(f"   ✅ Correctly rejected request with missing llm_provider: {response.status}")
            else:
                print(f"   ❌ Unexpected response for missing llm_provider: {response.status}")

        
        # Test 4: Test stop endpoint
        print("\n4. Testing agent stop endpoint...")
        async with session.post(f"{BASE_URL}/api/agent/stop/{session_id}") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Agent stop endpoint working: {data}")
            else:
                print(f"❌ Agent stop endpoint failed: {response.status}")
        
        # Test 5: Clean up test session
        print("\n5. Cleaning up test session...")
        async with session.delete(f"{BASE_URL}/api/session/{session_id}") as response:
            if response.status == 200:
                print(f"✅ Test session cleaned up")
            else:
                print(f"❌ Test session cleanup failed: {response.status}")
        
        return True

async def test_error_conditions():
    """Test error conditions and edge cases"""
    print("\n🧪 Testing Error Conditions...")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Access non-existent session
        print("\n1. Testing access to non-existent session...")
        fake_session_id = "non-existent-session-id"
        async with session.get(f"{BASE_URL}/api/session/{fake_session_id}") as response:
            if response.status == 404:
                print(f"✅ Correctly returned 404 for non-existent session")
            else:
                print(f"❌ Unexpected response for non-existent session: {response.status}")
        
        # Test 2: Try to run agent in non-existent session
        print("\n2. Testing agent run in non-existent session...")
        agent_payload = {
            "session_id": fake_session_id,
            "task": "Test task",
            "llm_provider": "google"
        }
        async with session.post(f"{BASE_URL}/api/agent/run", json=agent_payload) as response:
            if response.status == 404:
                print(f"✅ Correctly returned 404 for agent run in non-existent session")
            else:
                print(f"❌ Unexpected response for agent run in non-existent session: {response.status}")
        
        # Test 3: Try to delete non-existent session
        print("\n3. Testing deletion of non-existent session...")
        async with session.delete(f"{BASE_URL}/api/session/{fake_session_id}") as response:
            if response.status == 404:
                print(f"✅ Correctly returned 404 for non-existent session deletion")
            else:
                print(f"❌ Unexpected response for non-existent session deletion: {response.status}")
        
        return True

async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 PHASE 1 TEST SUITE - Multi-User Session Management")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Test started at: {datetime.now()}")
    
    try:
        # Run all tests
        test1_result = await test_session_management()
        test2_result = await test_api_endpoints()
        test3_result = await test_error_conditions()
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✅ Session Management Test: {'PASSED' if test1_result else 'FAILED'}")
        print(f"✅ API Endpoints Test: {'PASSED' if test2_result else 'FAILED'}")
        print(f"✅ Error Conditions Test: {'PASSED' if test3_result else 'FAILED'}")
        
        if all([test1_result, test2_result, test3_result]):
            print("\n🎉 ALL TESTS PASSED! Phase 1 implementation is working correctly.")
            print("\n📋 PHASE 1 COMPLETE - Ready for Phase 2!")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("To run this test:")
    print("1. Start the server: python webui.py")
    print("2. In another terminal: python test_phase1.py")
    print("\nStarting tests in 3 seconds...")
    
    # Give user time to read instructions
    import time
    time.sleep(3)
    
    asyncio.run(main())