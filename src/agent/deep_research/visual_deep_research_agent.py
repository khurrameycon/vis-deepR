# src/agent/deep_research/visual_deep_research_agent.py
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import base64

# Langchain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextWindowSize, BrowserContextConfig

# Import existing components - Fixed imports
from src.controller.custom_controller import CustomController
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import CustomBrowserContext
from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt

logger = logging.getLogger(__name__)

# Constants
REPORT_FILENAME = "research_report.md"
SEARCH_PROGRESS_FILE = "search_progress.json"

# Global tracking for sessions
_RESEARCH_SESSIONS = {}
_SESSION_BROWSERS = {}

class VisualDeepResearchAgent:
    """
    Visual Deep Research Agent with multi-user browser streaming support.
    
    Key changes from original:
    1. Each session gets its own browser instance
    2. Browser screenshots are streamed via WebSocket
    3. Research progress is tracked and displayed visually
    4. Minimal changes to existing research logic
    """
    
    def __init__(self, llm: Any, browser_config: Dict[str, Any], mcp_server_config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.browser_config = browser_config
        self.mcp_server_config = mcp_server_config
        self.mcp_client = None
        self.current_session_id: Optional[str] = None
        self.websocket_callback = None
        
    async def start_visual_research(
        self, 
        topic: str, 
        session_id: str,
        websocket_callback: callable,
        save_dir: str = "./tmp/visual_research",
        max_search_iterations: int = 5,
        max_parallel_browsers: int = 1
    ) -> Dict[str, Any]:
        """
        Start visual research with browser streaming.
        
        Args:
            topic: Research topic
            session_id: Unique session identifier  
            websocket_callback: Function to send updates via WebSocket
            save_dir: Directory to save research files
            max_search_iterations: Maximum research iterations
            max_parallel_browsers: Number of parallel browser instances
        """
        
        self.current_session_id = session_id
        self.websocket_callback = websocket_callback
        
        # Create session-specific directories
        session_dir = os.path.join(save_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Store session info
        _RESEARCH_SESSIONS[session_id] = {
            "topic": topic,
            "status": "starting",
            "progress": 0,
            "start_time": asyncio.get_event_loop().time(),
            "session_dir": session_dir
        }
        
        try:
            await self._send_update("status", {
                "message": f"🚀 Starting visual research on: {topic}",
                "progress": 0,
                "session_id": session_id
            })
            
            # Initialize browser for this session
            browser, browser_context = await self._setup_session_browser(session_id)
            
            # Start browser streaming
            streaming_task = asyncio.create_task(
                self._stream_browser_screenshots(session_id, browser_context)
            )
            
            # Run research with visual updates
            research_result = await self._run_visual_research_loop(
                topic, session_id, browser, browser_context, 
                max_search_iterations, session_dir
            )
            
            # Stop streaming
            streaming_task.cancel()
            
            await self._send_update("complete", {
                "message": "✅ Research completed successfully!",
                "progress": 100,
                "session_id": session_id,
                "result": research_result
            })
            
            return {
                "status": "completed",
                "session_id": session_id,
                "research_result": research_result,
                "files": {
                    "report": os.path.join(session_dir, REPORT_FILENAME),
                    "progress": os.path.join(session_dir, SEARCH_PROGRESS_FILE)
                }
            }
            
        except Exception as e:
            logger.error(f"Visual research failed for session {session_id}: {e}", exc_info=True)
            
            await self._send_update("error", {
                "message": f"❌ Research failed: {str(e)}",
                "session_id": session_id,
                "error": str(e)
            })
            
            return {
                "status": "failed", 
                "session_id": session_id,
                "error": str(e)
            }
            
        finally:
            # Cleanup session
            await self._cleanup_session(session_id)
    
    async def _setup_session_browser(self, session_id: str):
        """Setup browser instance for specific session"""
        
        await self._send_update("status", {
            "message": "🌐 Launching browser for your session...",
            "progress": 5,
            "session_id": session_id
        })
        
        # Browser configuration from existing setup
        headless = self.browser_config.get("headless", False)
        window_w = self.browser_config.get("window_width", 1280)
        window_h = self.browser_config.get("window_height", 1100)
        disable_security = self.browser_config.get("disable_security", True)
        browser_binary_path = self.browser_config.get("browser_binary_path", None)
        wss_url = self.browser_config.get("wss_url", None)
        cdp_url = self.browser_config.get("cdp_url", None)
        
        extra_args = [f"--window-size={window_w},{window_h}"]
        
        # Use the same configuration pattern as in webui.py
        browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                extra_chromium_args=extra_args,
                wss_url=wss_url,
                cdp_url=cdp_url,
            )
        )
        
        # Use the correct BrowserContextConfig class
        context_config = BrowserContextConfig(
            save_downloads_path=f"./tmp/downloads_{session_id}",
            browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
        )
        
        browser_context = await browser.new_context(config=context_config)
        
        # Store browser for this session
        _SESSION_BROWSERS[session_id] = {
            "browser": browser,
            "context": browser_context,
            "active": True
        }
        
        await self._send_update("status", {
            "message": "✅ Browser ready! Starting research...", 
            "progress": 10,
            "session_id": session_id
        })
        
        return browser, browser_context
    
    async def _stream_browser_screenshots(self, session_id: str, browser_context):
        """Stream browser screenshots via WebSocket"""
        
        logger.info(f"Starting screenshot streaming for session {session_id}")
        
        while session_id in _SESSION_BROWSERS and _SESSION_BROWSERS[session_id]["active"]:
            try:
                # Take screenshot
                screenshot_b64 = await self._capture_screenshot(browser_context)
                
                if screenshot_b64:
                    await self._send_update("screenshot", {
                        "session_id": session_id,
                        "image": screenshot_b64,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    logger.debug(f"Screenshot sent for session {session_id}")
                else:
                    logger.debug(f"No screenshot captured for session {session_id}")
                
                # Stream every 2 seconds (reduced frequency to avoid overwhelming)
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                logger.info(f"Screenshot streaming cancelled for session {session_id}")
                break
            except Exception as e:
                logger.debug(f"Screenshot capture failed for session {session_id}: {e}")
                await asyncio.sleep(5.0)  # Wait longer on errors
    
    async def _capture_screenshot(self, browser_context) -> Optional[str]:
        """Capture browser screenshot and return base64 encoded."""
        if not browser_context or not hasattr(browser_context, 'get_current_page'):
            logger.debug("Browser context not available or does not support get_current_page.")
            return None
        
        try:
            page = await browser_context.get_current_page()
            if page and not page.is_closed():
                screenshot_bytes = await page.screenshot(
                    type='jpeg',
                    quality=70,  # Optimized for streaming
                )
                return base64.b64encode(screenshot_bytes).decode('utf-8')
            else:
                logger.debug("No active page found for screenshot.")
        except Exception as e:
            logger.error(f"Screenshot capture failed with error: {e}", exc_info=True)
        
        return None
    
    async def _run_visual_research_loop(
        self, 
        topic: str, 
        session_id: str,
        browser, 
        browser_context,
        max_iterations: int,
        session_dir: str
    ) -> str:
        """
        Run the core research loop with visual updates.
        This is a simplified version of the original research logic.
        """
        
        # Initialize research tracking
        search_queries = []
        research_findings = []
        iteration_count = 0
        
        # Research planning phase
        await self._send_update("status", {
            "message": "🧠 Planning research approach...",
            "progress": 15,
            "session_id": session_id
        })
        
        # Generate initial research plan using LLM
        research_plan = await self._generate_research_plan(topic)
        
        await self._send_update("plan", {
            "message": f"📋 Research plan ready: {len(research_plan.get('queries', []))} searches planned",
            "progress": 20,
            "session_id": session_id,
            "plan": research_plan
        })
        
        # Main research loop
        for iteration in range(max_iterations):
            iteration_count = iteration + 1
            progress = 20 + (iteration / max_iterations) * 60  # 20% to 80% for research
            
            await self._send_update("status", {
                "message": f"🔍 Research iteration {iteration_count}/{max_iterations}",
                "progress": progress,
                "session_id": session_id,
                "iteration": iteration_count
            })
            
            # Get search queries for this iteration
            current_queries = await self._get_iteration_queries(
                topic, research_plan, search_queries, research_findings, iteration
            )
            
            if not current_queries:
                logger.info(f"No more queries for iteration {iteration_count}, research complete")
                break
                
            search_queries.extend(current_queries)
            
            # Execute searches with visual feedback
            for i, query in enumerate(current_queries):
                await self._send_update("search", {
                    "message": f"🔎 Searching: {query}",
                    "progress": progress + (i / len(current_queries)) * 10,
                    "session_id": session_id,
                    "query": query,
                    "iteration": iteration_count
                })
                
                # Execute search using existing CustomAgent
                search_result = await self._execute_visual_search(
                    query, browser, browser_context, session_id
                )
                
                if search_result:
                    research_findings.append({
                        "query": query,
                        "result": search_result,
                        "iteration": iteration_count,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                    
                    # Save progress
                    await self._save_research_progress(
                        session_dir, search_queries, research_findings
                    )
        
        # Generate final report
        await self._send_update("status", {
            "message": "📝 Generating research report...",
            "progress": 85,
            "session_id": session_id
        })
        
        final_report = await self._generate_final_report(
            topic, research_findings, session_dir
        )
        
        await self._send_update("status", {
            "message": "✅ Research report ready!",
            "progress": 100,
            "session_id": session_id
        })
        
        return final_report
    
    async def _generate_research_plan(self, topic: str) -> Dict[str, Any]:
        """Enhanced research plan generation for academic research"""
        
        plan_prompt = f"""You are a brilliant academic research assistant. Your task is to create a research plan for the topic: "{topic}"

        Generate 3-4 simple, keyword-based search queries. These queries should be effective on Google Scholar or similar academic search engines.
        **IMPORTANT: Do NOT use boolean operators like "AND", "OR", or parentheses. Use simple, direct keywords.**

        Return your response in this exact JSON format:
        {{
            "plan": "A concise academic research strategy.",
            "queries": ["simple query 1", "effective query 2", "another query"]
        }}"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=plan_prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            try:
                from json_repair import repair_json
                content = repair_json(content)
            except ImportError:
                logger.warning("json_repair not available, using standard json parsing")
            
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"Enhanced research plan generation failed: {e}")
            # Fallback academic plan
            return {
                "plan": f"Multi-source academic research on {topic}",
                "queries": [
                    f"{topic} fundamental concepts review",
                    f"{topic} recent advances 2023 2024", 
                    f"{topic} methodology survey",
                    f"{topic} applications case studies"
                ]
            }
    
    async def _get_iteration_queries(self, topic, research_plan, past_queries, findings, iteration) -> List[str]:
        """Get search queries for current iteration"""
        
        if iteration == 0:
            # First iteration - use planned queries
            return research_plan.get("queries", [])[:2]  # Limit to 2 queries per iteration
        
        # Subsequent iterations - generate based on findings
        if not findings:
            return []
            
        refine_prompt = f"""Based on current research findings, suggest 1-2 more specific search queries for: "{topic}"

Previous queries: {past_queries[-3:]}
Recent findings: {[f['result'][:200] + '...' for f in findings[-2:]]}

Generate focused queries to fill knowledge gaps. Return JSON:
{{"queries": ["specific_query1", "specific_query2"]}}"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=refine_prompt)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            
            # Try to import json_repair, fallback to json if not available
            try:
                from json_repair import repair_json
                content = repair_json(content)
            except ImportError:
                pass
            
            result = json.loads(content)
            return result.get("queries", [])[:2]  # Limit queries
            
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return []
    
    # async def _execute_visual_search(self, query: str, browser, browser_context, session_id: str) -> Optional[str]:
    #     """
    #     Enhanced academic search with multiple sources and deep paper reading.
    #     Priority: Semantic Scholar -> arXiv -> DuckDuckGo -> Google Scholar
    #     """
        
    #     search_results = []
    #     citations = []
        
    #     try:
    #         page = await browser_context.get_current_page()
            
    #         # Method 1: Semantic Scholar (Primary - No Captcha)
    #         semantic_result = await self._search_semantic_scholar(query, page)
    #         if semantic_result:
    #             search_results.extend(semantic_result['results'])
    #             citations.extend(semantic_result['citations'])
            
    #         # Method 2: arXiv Search (For recent papers)
    #         if len(search_results) < 3:
    #             arxiv_result = await self._search_arxiv(query, page)
    #             if arxiv_result:
    #                 search_results.extend(arxiv_result['results'])
    #                 citations.extend(arxiv_result['citations'])
            
    #         # Method 3: DuckDuckGo Academic Search (Fallback)
    #         if len(search_results) < 2:
    #             duckduckgo_result = await self._search_duckduckgo_academic(query, page)
    #             if duckduckgo_result:
    #                 search_results.extend(duckduckgo_result['results'])
    #                 citations.extend(duckduckgo_result['citations'])
            
    #         # Combine results and format response
    #         if search_results:
    #             formatted_result = self._format_academic_results(query, search_results, citations)
    #             return formatted_result
    #         else:
    #             return f"No academic results found for: {query}"
                
    #     except Exception as e:
    #         logger.error(f"Enhanced academic search failed for '{query}': {e}")
    #         return f"Academic search encountered an error for: {query}"

    async def _execute_visual_search(self, query: str, browser, browser_context, session_id: str) -> Optional[str]:
        """
        Executes a full academic research task for a single query using a dedicated CustomAgent.
        The agent will navigate, find a relevant paper, read it, and summarize the content.
        """
        await self._send_update("document_update", {
            "content": f"🔍 **Currently Searching:** {query}\n\n*Finding relevant academic papers...*",
            "section": f"search_{query}",
            "word_count": 0
        })
        
        
        try:
            # This is the new, clear, multi-step prompt that tells the agent WHAT to do.
            task_prompt = f"""
            **Academic Research Mission for query: "{query}"**

            Your task is to find and summarize a single, highly relevant academic paper.

            **Execution Plan:**
            1.  **Primary Search (Semantic Scholar):** First, navigate to `https://www.semanticscholar.org` and search for the query. If you find a relevant paper with a PDF or direct access link, click it.
            2.  **More Search (arXiv):** For more papers you should also navigate to `https://arxiv.org` and search. Prioritize recent papers.
            3.  **Fallback Search (DuckDuckGo):** If no paper is found on both sources, use DuckDuckGo to search for `"{query}" filetype:pdf` to find direct PDF documents.
            4.  **Search:** Execute the search query provided above.
            5.  **Analyze & Click:** Carefully examine the search results. Your primary goal is to find and click on a link that leads directly to a **[PDF]**. If no direct PDF link is available, click on the most promising and reputable academic source (e.g., arxiv.org, acm.org, ieee.org, springer.com).
            6.  **Read & Extract:** Once you have navigated to the paper, extract the key information, focusing on the abstract, introduction, and conclusion.
            7.  **Summarize:** Your final output must be a concise summary of the paper's content, including the **Title, Authors, and the Source URL** you used. Conclude your task after summarizing.
            """

            logger.info(f"Delegating deep research task for query: '{query}' to a dedicated CustomAgent.")
            
            # We create a new, focused agent for this specific research task.
            agent = CustomAgent(
                task=task_prompt,
                llm=self.llm,
                browser=browser,
                browser_context=browser_context,
                controller=CustomController(),
                use_vision=True,  # Vision is critical for identifying links.
                max_actions_per_step=7, # Allowing for more complex actions per step if needed
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt 
            )

            # Give the agent enough steps to complete the entire search-click-read-summarize cycle.
            
            result_history = await agent.run(max_steps=20) 
            final_data = result_history.final_result() or "The agent did not produce a final summary for this query."

            # Send paper summary (persistent - appends to literature section)
            await self._send_update("document_update", {
                "content": f"### Paper Found: {query}\n\n{final_data}\n\n---\n",
                "section": f"summary_{query}",
                "word_count": len(final_data.split())
            })
            
            logger.info(f"CustomAgent finished for query '{query}' with result.")
            return final_data

        except Exception as e:
            error_msg = f"**Search Error for '{query}':** {str(e)}"
            
            # Send error (persistent - appends to errors section)
            await self._send_update("document_update", {
                "content": f"❌ {error_msg}\n\n",
                "section": f"error_{query}",
                "word_count": 0
            })
            
            logger.error(f"The dedicated research agent failed for query '{query}': {e}", exc_info=True)
            return error_msg
    
    async def _send_agent_log(self, log_message: str, log_type: str = "info"):
        """Send agent step logs to frontend"""
        await self._send_update("agent_log", {
            "message": log_message,
            "type": log_type,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def _search_semantic_scholar(self, query: str, page) -> Optional[Dict]:
        """Search Semantic Scholar - Primary method (no captcha)"""
        try:
            # Use Semantic Scholar's search interface
            semantic_url = f"https://www.semanticscholar.org/search?q={query.replace(' ', '%20')}"
            await page.goto(semantic_url, wait_until="networkidle")
            await asyncio.sleep(3)
            
            # Extract paper results
            papers = await page.evaluate("""
                () => {
                    const results = [];
                    const citations = [];
                    
                    // Semantic Scholar paper cards
                    const paperCards = document.querySelectorAll('[data-test-id="search-result"]');
                    
                    for (let i = 0; i < Math.min(3, paperCards.length); i++) {
                        const card = paperCards[i];
                        
                        const titleEl = card.querySelector('[data-test-id="paper-title"] a');
                        const authorsEl = card.querySelector('[data-test-id="paper-authors"]');
                        const yearEl = card.querySelector('[data-test-id="paper-year"]');
                        const abstractEl = card.querySelector('[data-test-id="paper-abstract"]');
                        const citationsEl = card.querySelector('[data-test-id="paper-citations"]');
                        
                        if (titleEl) {
                            const title = titleEl.textContent.trim();
                            const link = titleEl.href;
                            const authors = authorsEl ? authorsEl.textContent.trim() : 'Unknown authors';
                            const year = yearEl ? yearEl.textContent.trim() : 'Unknown year';
                            const abstract = abstractEl ? abstractEl.textContent.trim() : 'No abstract available';
                            const citationCount = citationsEl ? citationsEl.textContent.trim() : '0';
                            
                            results.push({
                                title: title,
                                authors: authors,
                                year: year,
                                abstract: abstract.substring(0, 300) + '...',
                                link: link,
                                source: 'Semantic Scholar',
                                citations: citationCount
                            });
                            
                            // Format citation
                            citations.push(`${authors} (${year}). ${title}. Semantic Scholar. ${link}`);
                        }
                    }
                    
                    return { results, citations };
                }
            """)
            
            # Try to open and read first paper
            if papers['results']:
                enhanced_results = await self._deep_read_papers(papers['results'][:2], page)
                papers['results'] = enhanced_results
            
            return papers
            
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
            return None

    async def _search_arxiv(self, query: str, page) -> Optional[Dict]:
        """Search arXiv for recent papers"""
        try:
            # arXiv search interface
            arxiv_url = f"https://arxiv.org/search/?query={query.replace(' ', '+')}&searchtype=all"
            await page.goto(arxiv_url, wait_until="networkidle")
            await asyncio.sleep(2)
            
            papers = await page.evaluate("""
                () => {
                    const results = [];
                    const citations = [];
                    
                    const paperItems = document.querySelectorAll('li.arxiv-result');
                    
                    for (let i = 0; i < Math.min(2, paperItems.length); i++) {
                        const item = paperItems[i];
                        
                        const titleEl = item.querySelector('.list-title a');
                        const authorsEl = item.querySelector('.list-authors');
                        const abstractEl = item.querySelector('.list-summary .mathjax');
                        const subjectEl = item.querySelector('.list-subjects');
                        
                        if (titleEl) {
                            const title = titleEl.textContent.replace('Title: ', '').trim();
                            const link = 'https://arxiv.org' + titleEl.getAttribute('href');
                            const authors = authorsEl ? authorsEl.textContent.replace('Authors:', '').trim() : 'Unknown authors';
                            const abstract = abstractEl ? abstractEl.textContent.trim() : 'No abstract available';
                            const subjects = subjectEl ? subjectEl.textContent.replace('Subjects: ', '').trim() : '';
                            
                            results.push({
                                title: title,
                                authors: authors,
                                year: new Date().getFullYear().toString(), // arXiv is current
                                abstract: abstract.substring(0, 300) + '...',
                                link: link,
                                source: 'arXiv',
                                subjects: subjects
                            });
                            
                            citations.push(`${authors} (${new Date().getFullYear()}). ${title}. arXiv preprint. ${link}`);
                        }
                    }
                    
                    return { results, citations };
                }
            """)
            
            return papers
            
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return None

    async def _search_duckduckgo_academic(self, query: str, page) -> Optional[Dict]:
        """Search DuckDuckGo with academic operators"""
        try:
            # Academic search with site operators
            academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR filetype:pdf"
            duckduckgo_url = f"https://duckduckgo.com/?q={academic_query.replace(' ', '+')}"
            await page.goto(duckduckgo_url, wait_until="networkidle")
            await asyncio.sleep(2)
            
            papers = await page.evaluate("""
                () => {
                    const results = [];
                    const citations = [];
                    
                    const searchResults = document.querySelectorAll('[data-result]');
                    
                    for (let i = 0; i < Math.min(2, searchResults.length); i++) {
                        const result = searchResults[i];
                        
                        const titleEl = result.querySelector('h3 a');
                        const snippetEl = result.querySelector('.result__snippet');
                        const urlEl = result.querySelector('.result__url');
                        
                        if (titleEl && (titleEl.href.includes('arxiv') || titleEl.href.includes('pdf'))) {
                            const title = titleEl.textContent.trim();
                            const link = titleEl.href;
                            const snippet = snippetEl ? snippetEl.textContent.trim() : 'No description available';
                            
                            results.push({
                                title: title,
                                authors: 'Various authors',
                                year: 'Recent',
                                abstract: snippet.substring(0, 250) + '...',
                                link: link,
                                source: 'DuckDuckGo Academic'
                            });
                            
                            citations.push(`${title}. Retrieved from ${link}`);
                        }
                    }
                    
                    return { results, citations };
                }
            """)
            
            return papers
            
        except Exception as e:
            logger.warning(f"DuckDuckGo academic search failed: {e}")
            return None
    

    async def _deep_read_papers(self, papers: List[Dict], page) -> List[Dict]:
        """Deep read papers by opening their links and extracting content"""
        enhanced_papers = []
        
        for paper in papers:
            try:
                await self._send_update("status", {
                    "message": f"📖 Reading paper: {paper['title'][:50]}...",
                    "session_id": self.current_session_id
                })
                
                # Navigate to paper
                await page.goto(paper['link'], wait_until="networkidle")
                await asyncio.sleep(3)
                
                # Extract additional content based on source
                enhanced_content = await self._extract_paper_content(page, paper['source'])
                
                # Merge enhanced content with original paper info
                paper.update(enhanced_content)
                enhanced_papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Failed to deep read paper {paper['title']}: {e}")
                # Keep original paper info if deep reading fails
                enhanced_papers.append(paper)
        
        return enhanced_papers


    async def _extract_paper_content(self, page, source: str) -> Dict:
        """Extract content from different paper sources"""
        try:
            if source == "Semantic Scholar":
                return await page.evaluate("""
                    () => {
                        const abstract = document.querySelector('.paper-detail-abstract .text-gray-700');
                        const doi = document.querySelector('[data-test-id="paper-doi"] a');
                        const venue = document.querySelector('[data-test-id="paper-venue"]');
                        const references = document.querySelectorAll('.paper-references .citation');
                        
                        return {
                            full_abstract: abstract ? abstract.textContent.trim() : null,
                            doi: doi ? doi.href : null,
                            venue: venue ? venue.textContent.trim() : null,
                            reference_count: references.length,
                            content_extracted: true
                        };
                    }
                """)
                
            elif source == "arXiv":
                return await page.evaluate("""
                    () => {
                        const abstract = document.querySelector('.abstract .mathjax');
                        const pdfLink = document.querySelector('.download-pdf');
                        const comments = document.querySelector('.comments .mathjax');
                        
                        return {
                            full_abstract: abstract ? abstract.textContent.replace('Abstract: ', '').trim() : null,
                            pdf_link: pdfLink ? pdfLink.href : null,
                            comments: comments ? comments.textContent.trim() : null,
                            content_extracted: true
                        };
                    }
                """)
            
            else:
                # Generic content extraction
                return await page.evaluate("""
                    () => {
                        const content = document.querySelector('main, article, .content');
                        return {
                            page_content: content ? content.textContent.substring(0, 500) + '...' : null,
                            content_extracted: true
                        };
                    }
                """)
                
        except Exception as e:
            logger.warning(f"Content extraction failed for {source}: {e}")
            return {"content_extracted": False}
    
    def _format_academic_results(self, query: str, results: List[Dict], citations: List[str]) -> str:
        """Format academic search results with proper citations"""
        
        formatted_output = f"Academic Research Results for '{query}':\n\n"
        
        for i, paper in enumerate(results, 1):
            formatted_output += f"**Paper {i}: {paper['title']}**\n"
            formatted_output += f"Authors: {paper['authors']}\n"
            formatted_output += f"Year: {paper['year']}\n"
            formatted_output += f"Source: {paper['source']}\n"
            
            if paper.get('venue'):
                formatted_output += f"Venue: {paper['venue']}\n"
            
            if paper.get('full_abstract'):
                formatted_output += f"Abstract: {paper['full_abstract']}\n"
            else:
                formatted_output += f"Abstract: {paper['abstract']}\n"
            
            if paper.get('doi'):
                formatted_output += f"DOI: {paper['doi']}\n"
            
            formatted_output += f"Link: {paper['link']}\n"
            formatted_output += "-" * 80 + "\n\n"
        
        # Add citations section
        if citations:
            formatted_output += "\n**REFERENCES:**\n"
            for i, citation in enumerate(citations, 1):
                formatted_output += f"[{i}] {citation}\n"
        
        return formatted_output

    async def _generate_final_report(self, topic: str, findings: List[Dict], session_dir: str) -> str:
        """Generate a final academic report with proper citations."""
        
        if not findings:
            return f"# Research Report: {topic}\n\nNo research findings were gathered during the process."

        # Send initial document structure
        await self._send_update("document_update", {
            "content": f"# Research Report: {topic}\n\n*Generating report...*",
            "section": "initialization",
            "word_count": 0
        })

        findings_text = ""
        citations = []
        
        # Process findings to extract results and citations
        for i, finding in enumerate(findings):
            query = finding['query']
            result_content = finding['result']
            
            # Send progress update
            await self._send_update("document_update", {
                "content": f"# Research Report: {topic}\n\n*Processing finding {i+1}/{len(findings)}...*\n\n## Findings So Far:\n{findings_text}",
                "section": "processing",
                "word_count": len(findings_text.split())
            })
            
            # Extract citations from the formatted result string
            if "**REFERENCES:**" in result_content:
                parts = result_content.split("**REFERENCES:**")
                main_content = parts[0]
                references_section = parts[1].strip().split('\n')
                citations.extend(references_section)
            else:
                main_content = result_content

            findings_text += f"### Research from Query: \"{query}\"\n\n{main_content.strip()}\n\n---\n"

        # Create a unique, numbered list of citations
        unique_citations = sorted(list(set(c for c in citations if c.strip())))
        
        # Send findings completion
        await self._send_update("document_update", {
            "content": f"# Research Report: {topic}\n\n*Synthesizing final report...*\n\n## Research Findings:\n{findings_text}",
            "section": "findings_complete",
            "word_count": len(findings_text.split())
        })
        
        # Prepare the research plan context
        plan_summary = "\nResearch Plan Followed:\n"
        for item in self.research_plan if hasattr(self, 'research_plan') else []:
            marker = "- [x]" if item.get('status') == 'completed' else "- [ ] (Failed)" if item.get('status') == 'failed' else "- [ ]"
            plan_summary += f"{marker} {item['task']}\n"

        report_prompt = f"""
        **Task**: Generate a professional, well-structured academic research report on the topic: "{topic}".

        **Instructions**:
        1. Synthesize the provided research findings into a coherent report with an Introduction, Detailed Analysis, and Conclusion.
        2. Analyze the extracted content to build your report. Do not simply list the findings.
        3. **You must base your report *only* on the information provided below.**

        **Aggregated Research Findings**:
        {findings_text}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=report_prompt)])
            report_content = response.content
            
            # Stream the final report as it's generated
            await self._send_update("document_update", {
                "content": report_content,
                "section": "final_report",
                "word_count": len(report_content.split()),
                "status": "complete"
            })
            
            # Append the cleaned and numbered reference list
            if unique_citations:
                report_content += "\n\n## References\n"
                for i, citation in enumerate(unique_citations, 1):
                    if citation.startswith(f"[{i}]"):
                        report_content += f"{citation}\n"
                    else:
                        report_content += f"[{i}] {citation.lstrip('[]0123456789. ')}\n"

            # Send final complete document
            await self._send_update("document_update", {
                "content": report_content,
                "section": "complete",
                "word_count": len(report_content.split()),
                "status": "final"
            })
            
            # Save the final report
            report_path = os.path.join(session_dir, REPORT_FILENAME)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Final academic report saved to: {report_path}")
            return report_content
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            error_report = f"# Report Generation Failed\n\nAn error occurred during the final synthesis: {str(e)}"
            
            await self._send_update("document_update", {
                "content": error_report,
                "section": "error",
                "word_count": len(error_report.split()),
                "status": "error"
            })
            
            return error_report
    
    async def _save_research_progress(self, session_dir: str, queries: List[str], findings: List[Dict]):
        """Save research progress to JSON file"""
        try:
            progress_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "total_queries": len(queries),
                "total_findings": len(findings),
                "queries": queries,
                "findings": [
                    {
                        "query": f["query"],
                        "result_length": len(f["result"]),
                        "iteration": f["iteration"],
                        "timestamp": f["timestamp"]
                    }
                    for f in findings
                ]
            }
            
            progress_path = os.path.join(session_dir, SEARCH_PROGRESS_FILE)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    async def _send_update(self, update_type: str, data: Dict[str, Any]):
        """Send update via WebSocket callback with the correct structure for the frontend."""
        if self.websocket_callback:
            try:
                # The agent will now format the message directly for the frontend
                message_to_send = {
                    "type": "research_update",
                    "data": {
                        "type": update_type,
                        **data
                    }
                }
                
                await self.websocket_callback(message_to_send)
            except Exception as e:
                logger.debug(f"WebSocket update failed: {e}")
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources"""
        try:
            # Close browser resources
            if session_id in _SESSION_BROWSERS:
                browser_info = _SESSION_BROWSERS[session_id]
                browser_info["active"] = False
                
                if browser_info.get("context"):
                    await browser_info["context"].close()
                if browser_info.get("browser"):
                    await browser_info["browser"].close()
                
                del _SESSION_BROWSERS[session_id]
            
            # Clean up session tracking
            if session_id in _RESEARCH_SESSIONS:
                del _RESEARCH_SESSIONS[session_id]
            
            # Close MCP client if exists
            if self.mcp_client:
                await self.mcp_client.__aexit__(None, None, None)
                self.mcp_client = None
                
            logger.info(f"Session {session_id} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Session cleanup failed for {session_id}: {e}")
    
    async def stop_research(self, session_id: str):
        """Stop research for specific session"""
        if session_id in _SESSION_BROWSERS:
            _SESSION_BROWSERS[session_id]["active"] = False
        
        if session_id in _RESEARCH_SESSIONS:
            _RESEARCH_SESSIONS[session_id]["status"] = "stopped"
        
        await self._cleanup_session(session_id)
        
        logger.info(f"Research stopped for session: {session_id}")
    
    @classmethod
    def get_session_info(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about research session"""
        return _RESEARCH_SESSIONS.get(session_id)
    
    @classmethod 
    def get_active_sessions(cls) -> List[str]:
        """Get list of active research session IDs"""
        return list(_RESEARCH_SESSIONS.keys())