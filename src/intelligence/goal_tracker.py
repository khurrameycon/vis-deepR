import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GoalTracker:
    """Tracks main goal progress and detects deviations"""
    
    def __init__(self, main_goal: str):
        self.main_goal = main_goal
        self.sub_goals = []
        self.current_progress = 0
        self.recent_states = []  # Track last 5 states
        self.stuck_patterns = []
        self.deviation_count = 0
        self.goal_keywords = self._extract_keywords(main_goal)
        
    def _extract_keywords(self, goal: str) -> List[str]:
        """Extract key terms from the main goal"""
        import re
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', goal.lower())
        # Filter out common words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with'}
        return [w for w in words if len(w) > 2 and w not in stop_words]
    
    async def decompose_goal(self, llm) -> List[Dict]:
        """Break down main goal into sub-goals with LLM"""
        decomposition_prompt = f"""Break down this main goal into logical sub-goals:

MAIN GOAL: {self.main_goal}

Create 3-6 sub-goals that represent the logical steps to achieve this goal.
Each sub-goal should be specific and measurable.

Respond in JSON:
{{"sub_goals": [
  {{"id": 1, "description": "...", "status": "pending", "priority": "high/medium/low"}},
  {{"id": 2, "description": "...", "status": "pending", "priority": "high/medium/low"}}
]}}"""

        try:
            response = await llm.ainvoke([{"role": "user", "content": decomposition_prompt}])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            from json_repair import repair_json
            content = repair_json(content)
            result = json.loads(content)
            
            self.sub_goals = result.get('sub_goals', [])
            logger.info(f"🎯 Goal decomposed into {len(self.sub_goals)} sub-goals")
            return self.sub_goals
            
        except Exception as e:
            logger.warning(f"Goal decomposition failed: {e}")
            # Fallback decomposition
            self.sub_goals = [
                {"id": 1, "description": f"Complete: {self.main_goal}", "status": "pending", "priority": "high"}
            ]
            return self.sub_goals
    
    async def check_alignment(self, current_state: Dict, step_description: str, llm) -> Dict[str, Any]:
        """Check if current action aligns with main goal"""
        
        alignment_prompt = f"""Analyze goal alignment:

MAIN GOAL: {self.main_goal}
KEY TERMS: {', '.join(self.goal_keywords)}

CURRENT SITUATION:
- URL: {current_state.get('current_url', 'unknown')}
- Step: {step_description}
- Recent Progress: {self._get_recent_progress_summary()}

ANALYSIS NEEDED:
1. Is this step directly contributing to the main goal?
2. Are we on the right track or getting distracted?
3. What percentage of the main goal is complete?
4. What should be the immediate next focus?

Respond in JSON:
{{"alignment": "on_track/minor_deviation/major_deviation", "confidence": 0.8, "progress_percentage": 75, "reasoning": "...", "next_focus": "..."}}"""

        try:
            response = await llm.ainvoke([{"role": "user", "content": alignment_prompt}])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            from json_repair import repair_json
            content = repair_json(content)
            alignment = json.loads(content)
            
            # Update progress
            self.current_progress = alignment.get('progress_percentage', self.current_progress)
            
            logger.info(f"🎯 Goal alignment: {alignment.get('alignment')} ({alignment.get('confidence', 0):.1f} confidence)")
            return alignment
            
        except Exception as e:
            logger.warning(f"Goal alignment check failed: {e}")
            return {
                "alignment": "on_track",
                "confidence": 0.5,
                "progress_percentage": self.current_progress,
                "reasoning": "Could not analyze alignment",
                "next_focus": "Continue with current step"
            }
    
    async def detect_stuck_pattern(self, recent_actions: List[Dict], llm) -> Optional[Dict]:
        """Detect if agent is stuck in a loop or pattern"""
        
        if len(recent_actions) < 3:
            return None
        
        # Simple pattern detection
        last_3_actions = [a.get('action', '') for a in recent_actions[-3:]]
        last_3_urls = [a.get('url', '') for a in recent_actions[-3:]]
        
        # Check for repeated actions
        if len(set(last_3_actions)) == 1:  # Same action 3 times
            stuck_analysis_prompt = f"""Agent appears stuck - same action repeated:

REPEATED ACTION: {last_3_actions[0]}
URLS: {last_3_urls}
MAIN GOAL: {self.main_goal}

ANALYSIS:
1. Why might this action be failing?
2. What alternative approaches could work?
3. Should we skip this step or try a different method?

Respond in JSON:
{{"stuck": true, "reason": "...", "solutions": ["solution1", "solution2"], "recommended_action": "retry/skip/alternative"}}"""

            try:
                response = await llm.ainvoke([{"role": "user", "content": stuck_analysis_prompt}])
                content = response.content.replace("```json", "").replace("```", "").strip()
                
                from json_repair import repair_json
                content = repair_json(content)
                stuck_analysis = json.loads(content)
                
                logger.warning(f"🔄 Stuck pattern detected: {stuck_analysis.get('reason')}")
                return stuck_analysis
                
            except Exception as e:
                logger.warning(f"Stuck pattern analysis failed: {e}")
                return {
                    "stuck": True,
                    "reason": "Repeated action detected",
                    "solutions": ["Try different approach", "Skip this step"],
                    "recommended_action": "alternative"
                }
        
        return None
    
    async def suggest_course_correction(self, deviation_reason: str, current_state: Dict, llm) -> Dict[str, Any]:
        """Suggest how to get back on track"""
        
        correction_prompt = f"""Agent has deviated from goal. Suggest course correction:

MAIN GOAL: {self.main_goal}
DEVIATION REASON: {deviation_reason}
CURRENT STATE:
- URL: {current_state.get('current_url', 'unknown')}
- Progress: {self.current_progress}%

SUB-GOALS STATUS:
{json.dumps(self.sub_goals, indent=2)}

CORRECTION NEEDED:
1. What's the fastest way back to the main goal?
2. What specific action should be taken?
3. Should any steps be skipped or modified?

Respond in JSON:
{{"correction_type": "navigate_back/search_again/skip_step/refocus", "action": "...", "reasoning": "...", "urgency": "high/medium/low"}}"""

        try:
            response = await llm.ainvoke([{"role": "user", "content": correction_prompt}])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            from json_repair import repair_json
            content = repair_json(content)
            correction = json.loads(content)
            
            self.deviation_count += 1
            logger.info(f"🔄 Course correction: {correction.get('action')}")
            return correction
            
        except Exception as e:
            logger.warning(f"Course correction failed: {e}")
            return {
                "correction_type": "refocus",
                "action": "Return to main goal focus",
                "reasoning": "Generic course correction",
                "urgency": "medium"
            }
    
    def update_state(self, url: str, action: str, result: str):
        """Update recent state tracking"""
        state_entry = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "action": action,
            "result": result[:200]  # Truncate long results
        }
        
        self.recent_states.append(state_entry)
        
        # Keep only last 5 states
        if len(self.recent_states) > 5:
            self.recent_states = self.recent_states[-5:]
    
    def _get_recent_progress_summary(self) -> str:
        """Get summary of recent progress"""
        if not self.recent_states:
            return "No recent progress"
        
        recent = self.recent_states[-3:]  # Last 3 states
        summary = []
        for state in recent:
            summary.append(f"• {state['action']} → {state['result'][:50]}...")
        
        return "\n".join(summary)
    
    def update_sub_goal_status(self, sub_goal_id: int, status: str):
        """Update status of a specific sub-goal"""
        for goal in self.sub_goals:
            if goal['id'] == sub_goal_id:
                goal['status'] = status
                logger.info(f"📋 Sub-goal {sub_goal_id} updated to: {status}")
                break
    
    def get_goal_summary(self) -> Dict[str, Any]:
        """Get complete goal tracking summary"""
        completed_sub_goals = len([g for g in self.sub_goals if g.get('status') == 'completed'])
        total_sub_goals = len(self.sub_goals)
        
        return {
            "main_goal": self.main_goal,
            "progress_percentage": self.current_progress,
            "sub_goals_completed": f"{completed_sub_goals}/{total_sub_goals}",
            "deviation_count": self.deviation_count,
            "recent_states_count": len(self.recent_states),
            "goal_keywords": self.goal_keywords
        }
