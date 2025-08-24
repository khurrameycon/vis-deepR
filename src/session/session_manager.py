# src/session/session_manager.py - Updated for Visual Research Support

import uuid
import asyncio
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fastapi import WebSocket

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Enhanced UserSession with research support"""
    session_id: str
    browser: Optional['CustomBrowser'] = None
    browser_context: Optional['CustomBrowserContext'] = None
    agent: Optional['CustomAgent'] = None
    current_task: Optional[asyncio.Task] = None
    websockets: List[WebSocket] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    execution_context: Dict = field(default_factory=dict)
    
    # Existing fields
    persistent: bool = False
    executed_steps: List[Dict] = field(default_factory=list)
    step_memory: str = ""
    current_step_index: int = 0
    stopped: bool = False
    goal_tracker: Optional['GoalTracker'] = None
    
    # NEW: Research-specific fields
    research_task: Optional[asyncio.Task] = None
    research_agent: Optional['VisualDeepResearchAgent'] = None
    research_active: bool = False
    research_topic: Optional[str] = None
    research_progress: int = 0
    research_start_time: Optional[datetime] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def start_research(self, topic: str, research_agent):
        """Mark session as having active research"""
        self.research_active = True
        self.research_topic = topic
        self.research_agent = research_agent
        self.research_start_time = datetime.now()
        self.research_progress = 0
        self.update_activity()
    
    def update_research_progress(self, progress: int):
        """Update research progress"""
        self.research_progress = max(0, min(100, progress))
        self.update_activity()
    
    def complete_research(self):
        """Mark research as completed"""
        self.research_active = False
        self.research_progress = 100
        self.research_task = None
        self.research_agent = None
        self.update_activity()
    
    async def cleanup(self):
        """Clean up session resources including research"""
        try:
            # Cancel research task if running
            if self.research_task and not self.research_task.done():
                logger.info(f"Cancelling research task for session {self.session_id}")
                self.research_task.cancel()
                try:
                    await self.research_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling research task for session {self.session_id}: {e}")
            
            # Clean up research agent
            if self.research_agent:
                try:
                    await self.research_agent.stop_research(self.session_id)
                except Exception as e:
                    logger.error(f"Error stopping research agent for session {self.session_id}: {e}")
                self.research_agent = None
            
            # Cancel regular agent task
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task for session {self.session_id}: {e}")
            
            # Only close browser if not persistent
            if not self.persistent:
                if self.browser_context:
                    await self.browser_context.close()
                    self.browser_context = None
                
                if self.browser:
                    await self.browser.close()
                    self.browser = None
            else:
                logger.info(f"Session {self.session_id} is persistent - keeping browser alive")
            
            # Close WebSocket connections
            for ws in self.websockets:
                try:
                    await ws.close()
                except Exception:
                    pass
            self.websockets.clear()
            
            logger.info(f"Session {self.session_id} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {self.session_id}: {e}")

class SessionManager:
    def __init__(self, session_timeout_minutes: int = 45, max_sessions: int = 20):
        self.sessions: Dict[str, UserSession] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions = max_sessions
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        # Check if we've reached max sessions
        if len(self.sessions) >= self.max_sessions:
            # Clean up expired sessions first
            asyncio.create_task(self.cleanup_expired_sessions())
            if len(self.sessions) >= self.max_sessions:
                raise Exception(f"Maximum number of sessions ({self.max_sessions}) reached")
        
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = UserSession(session_id=session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID and update activity"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_activity()
            return session
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            await session.cleanup()
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    async def cleanup_expired_sessions(self):
        """Clean up sessions that have expired (skip persistent and research sessions)"""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            # Skip persistent sessions from auto-cleanup
            if session.persistent:
                continue
                
            # Skip active research sessions from auto-cleanup
            if session.research_active:
                # But check if research has been stuck for too long (2 hours)
                if (session.research_start_time and 
                    now - session.research_start_time > timedelta(hours=2)):
                    logger.warning(f"Research session {session_id} stuck for >2 hours, cleaning up")
                    expired_sessions.append(session_id)
                continue
            
            # Regular timeout check
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            await self.delete_session(session_id)
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        return len(self.sessions)
    
    def get_research_sessions(self) -> List[Dict[str, any]]:
        """Get all active research sessions"""
        research_sessions = []
        for session_id, session in self.sessions.items():
            if session.research_active:
                research_sessions.append({
                    "session_id": session_id,
                    "topic": session.research_topic,
                    "progress": session.research_progress,
                    "start_time": session.research_start_time.isoformat() if session.research_start_time else None,
                    "duration_minutes": (
                        (datetime.now() - session.research_start_time).total_seconds() / 60
                        if session.research_start_time else 0
                    ),
                    "websocket_count": len(session.websockets)
                })
        return research_sessions
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get comprehensive session information including research status"""
        session = self.get_session(session_id)
        if session:
            info = {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "has_browser": session.browser is not None,
                "has_context": session.browser_context is not None,
                "has_agent": session.agent is not None,
                "task_running": session.current_task is not None and not session.current_task.done(),
                "websocket_count": len(session.websockets),
                "persistent": session.persistent,
                # Research-specific info
                "research_active": session.research_active,
                "research_topic": session.research_topic,
                "research_progress": session.research_progress,
                "research_start_time": session.research_start_time.isoformat() if session.research_start_time else None,
                "research_task_running": session.research_task is not None and not session.research_task.done(),
            }
            return info
        return None
    
    async def shutdown(self):
        """Shutdown session manager and clean up all sessions"""
        logger.info("Shutting down session manager...")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all sessions (including research sessions)
        session_ids = list(self.sessions.keys())
        cleanup_tasks = []
        
        for session_id in session_ids:
            cleanup_tasks.append(self.delete_session(session_id))
        
        # Wait for all sessions to cleanup
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("Session manager shutdown complete")