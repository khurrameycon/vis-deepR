import pdb
import asyncio
import pyperclip
from typing import Optional, Type
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    # def _register_custom_actions(self):
    #     """Register all custom browser actions"""

    #     @self.registry.action("Copy text to clipboard")
    #     def copy_to_clipboard(text: str):
    #         pyperclip.copy(text)
    #         return ActionResult(extracted_content=text)

    #     @self.registry.action("Paste text from clipboard")
    #     async def paste_from_clipboard(browser: BrowserContext):
    #         text = pyperclip.paste()
    #         # send text to browser
    #         page = await browser.get_current_page()
    #         await page.keyboard.type(text)

    #         return ActionResult(extracted_content=text)

    #     @self.registry.action("Smart scroll up when stuck")
    #     async def smart_scroll_up(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         # Scroll to top of page
    #         await page.keyboard.press('Home')
    #         await asyncio.sleep(1)
    #         return ActionResult(extracted_content="Scrolled to top to recover from stuck state")
        
    #     @self.registry.action("Close popup or modal")
    #     async def close_popup(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         # Try common popup close selectors
    #         selectors = ['[aria-label="Close"]', '.close', '.modal-close', '[data-dismiss="modal"]', 'button:has-text("Close")', 'button:has-text("×")']
    #         for selector in selectors:
    #             try:
    #                 await page.click(selector, timeout=2000)
    #                 return ActionResult(extracted_content="Closed popup/modal")
    #             except:
    #                 continue
    #         # Try Escape key
    #         await page.keyboard.press('Escape')
    #         return ActionResult(extracted_content="Pressed Escape to close popup")
        
    #     @self.registry.action("Detect and handle stuck state")
    #     async def handle_stuck_state(browser: BrowserContext):
    #         page = await browser.get_current_page()
    #         current_url = page.url
    #         # Check if in footer or problematic area
    #         try:
    #             footer_element = await page.query_selector('footer, .footer, #footer')
    #             if footer_element:
    #                 await page.keyboard.press('Home')  # Go to top
    #                 return ActionResult(extracted_content="Detected footer area, scrolled to top")
    #         except:
    #             pass
    #         return ActionResult(extracted_content="State check completed")
    
    def _register_custom_actions(self):
        """Register all custom browser actions"""
        
        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)
    
        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            page = await browser.get_current_page()
            await page.keyboard.type(text)
            return ActionResult(extracted_content=text)
        
        @self.registry.action("Close popup or dismiss overlay")
        async def close_popup(browser: BrowserContext):
            page = await browser.get_current_page()
            closed_something = False
            
            # Try specific popup close patterns
            close_patterns = [
                # Common close buttons
                'button[aria-label*="close" i]',
                'button[aria-label*="dismiss" i]',
                '[data-testid*="close"]',
                '[class*="close"]',
                '[class*="dismiss"]',
                # Modal specific
                '.modal-close',
                '.overlay-close',
                # X buttons
                'button:has-text("×")',
                'button:has-text("✕")',
                'span:has-text("×")',
                # Cookie/GDPR buttons
                'button:has-text("Accept")',
                'button:has-text("OK")',
                'button:has-text("Got it")',
                'button:has-text("Continue")',
                # Generic buttons that might close popups
                '[role="button"][aria-label*="close" i]'
            ]
            
            for pattern in close_patterns:
                try:
                    elements = await page.query_selector_all(pattern)
                    for element in elements:
                        if await element.is_visible():
                            await element.click(timeout=1000)
                            closed_something = True
                            await asyncio.sleep(0.5)
                            break
                    if closed_something:
                        break
                except:
                    continue
            
            # Try Escape key as backup
            if not closed_something:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
            
            return ActionResult(extracted_content="Attempted to close popup/overlay")
        
        @self.registry.action("Scroll to top of page")
        async def scroll_to_top(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.keyboard.press('Home')
            await asyncio.sleep(1)
            return ActionResult(extracted_content="Scrolled to top of page")
        
        @self.registry.action("Go to DuckDuckGo search")
        async def go_to_duckduckgo(browser: BrowserContext):
            page = await browser.get_current_page()
            await page.goto('https://duckduckgo.com')
            await asyncio.sleep(2)
            
            # Close any popups that might appear
            try:
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
            except:
                pass
            
            return ActionResult(extracted_content="Navigated to DuckDuckGo search engine")
    
        @self.registry.action("Search directly on DuckDuckGo with query")
        async def search_duckduckgo(query: str, browser: BrowserContext):
            """Search directly on DuckDuckGo using URL parameters to skip intro page"""
            import urllib.parse
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://duckduckgo.com/?t=h_&q={encoded_query}"
            
            page = await browser.get_current_page()
            await page.goto(search_url)
            await asyncio.sleep(2)  # Wait for page to load
            
            return ActionResult(extracted_content=f"Searched DuckDuckGo for: {query}")
