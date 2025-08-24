import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .goal_tracker import GoalTracker
logger = logging.getLogger(__name__)

class StepAnalyzer:
    """Analyzes steps before and after execution for intelligent behavior"""
    
    @staticmethod
    async def analyze_before_step(step: Dict, context: Dict, llm, goal_tracker: GoalTracker = None) -> Dict[str, Any]:
        """Analyze what the step should accomplish and how"""
        
        analysis_prompt = f"""Analyze this browser automation step:
    
    STEP TO EXECUTE:
    - Action: {step.get('action')}
    - Description: {step.get('description')}
    - Parameters: {json.dumps(step.get('params', {}))}
    
    CURRENT CONTEXT:
    - URL: {context.get('current_url', 'unknown')}
    - Previous results: {context.get('memory_chain', '')[-200:]}
    
    MAIN GOAL: {goal_tracker.main_goal if goal_tracker else 'Not specified'}
    
    ANALYZE:
    1. What is the goal of this step?
    2. How does this contribute to the main goal?
    3. What should I expect to see on the page?
    4. What could go wrong?
    5. How will I know if it succeeded?
    
    Respond in JSON:
    {{"goal": "...", "goal_contribution": "...", "expectations": "...", "risks": "...", "success_criteria": "..."}}"""
    
        try:
            response = await llm.ainvoke([{"role": "user", "content": analysis_prompt}])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            from json_repair import repair_json
            content = repair_json(content)
            analysis = json.loads(content)
            
            # Add goal alignment check
            if goal_tracker:
                alignment = await goal_tracker.check_alignment(
                    context, 
                    step.get('description', ''), 
                    llm
                )
                analysis['goal_alignment'] = alignment
            
            logger.info(f"🧠 Step Analysis: {analysis.get('goal', 'Unknown goal')}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Step analysis failed: {e}")
            return {
                "goal": "Execute step", 
                "goal_contribution": "Unknown",
                "expectations": "Unknown", 
                "risks": "Unknown", 
                "success_criteria": "Step completes"
            }
    
    @staticmethod
    async def analyze_after_step(step: Dict, result: str, context: Dict, llm, goal_tracker: GoalTracker = None) -> Dict[str, Any]:
        """Analyze if the step succeeded and what changed"""
        
        validation_prompt = f"""Evaluate this completed browser step:
    
    STEP EXECUTED:
    - Action: {step.get('action')}
    - Goal: {step.get('description')}
    - Result: {result[:300]}
    
    CONTEXT:
    - Current URL: {context.get('current_url', 'unknown')}
    
    MAIN GOAL: {goal_tracker.main_goal if goal_tracker else 'Not specified'}
    GOAL PROGRESS: {goal_tracker.current_progress if goal_tracker else 0}%
    
    EVALUATE:
    1. Did the step achieve its goal? (success/partial/failed)
    2. What actually happened?
    3. What information was gained?
    4. How much closer are we to the main goal?
    5. Should we continue as planned or adapt?
    
    Respond in JSON:
    {{"status": "success/partial/failed", "what_happened": "...", "data_extracted": {{}}, "goal_progress_delta": 10, "next_recommendation": "..."}}"""
    
        try:
            response = await llm.ainvoke([{"role": "user", "content": validation_prompt}])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            from json_repair import repair_json
            content = repair_json(content)
            analysis = json.loads(content)
            
            # Update goal tracker
            if goal_tracker:
                goal_tracker.update_state(
                    context.get('current_url', ''),
                    step.get('action', ''),
                    result
                )
                
                # Check for stuck patterns
                stuck_analysis = await goal_tracker.detect_stuck_pattern(
                    goal_tracker.recent_states,
                    llm
                )
                if stuck_analysis:
                    analysis['stuck_pattern'] = stuck_analysis
            
            logger.info(f"✅ Step Result: {analysis.get('status', 'unknown')} - {analysis.get('what_happened', '')[:100]}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Step validation failed: {e}")
            return {
                "status": "unknown", 
                "what_happened": result, 
                "data_extracted": {}, 
                "goal_progress_delta": 0,
                "next_recommendation": "continue"
            }
class AdaptivePlanner:
    """Handles dynamic step modification based on results"""
    
    @staticmethod
    async def should_adapt_plan(step_analysis: Dict, remaining_steps: List[Dict], llm) -> Optional[Dict]:
        """Determine if plan needs modification"""
        
        if step_analysis.get('status') == 'success':
            return None  # No adaptation needed
            
        if step_analysis.get('status') == 'failed':
            adaptation_prompt = f"""The current step failed. Suggest adaptations:

FAILED STEP: {step_analysis.get('what_happened', '')}
REMAINING STEPS: {json.dumps([s.get('description') for s in remaining_steps[:3]])}

OPTIONS:
1. Retry with different approach
2. Skip this step and continue  
3. Modify next steps
4. Add new intermediate step

Respond in JSON:
{{"action": "retry/skip/modify/add_step", "reason": "...", "new_step": {{}}, "modifications": []}}"""

            try:
                response = await llm.ainvoke([{"role": "user", "content": adaptation_prompt}])
                content = response.content.replace("```json", "").replace("```", "").strip()
                
                from json_repair import repair_json
                content = repair_json(content)
                adaptation = json.loads(content)
                
                logger.info(f"🔄 Adaptation: {adaptation.get('action')} - {adaptation.get('reason')}")
                return adaptation
                
            except Exception as e:
                logger.warning(f"Adaptation planning failed: {e}")
                return None
        
        return None

