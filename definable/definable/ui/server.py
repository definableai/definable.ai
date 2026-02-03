"""FastAPI server for UI with Jinja2 template rendering (Django-style)."""

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from definable.ui.chat import ChatWindow


class UIServer:
    """FastAPI server for serving the chat UI with Jinja2 template rendering (Django-style)."""

    def __init__(self, chat_window, port: int = 8000, cdn_url: str = None):
        self.chat_window: ChatWindow = chat_window
        self.port = port
        self.cdn_url = cdn_url
        self.app = FastAPI(title="Definable UI")
        
        # Setup Jinja2 templates (Django-style)
        templates_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))

        # Add CORS middleware to allow cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_root(request: Request):
            # Render template with context (Django-style)
            return self.templates.TemplateResponse(
                "chat.html",
                {
                    "request": request,
                    "title": self.chat_window.title,
                    "theme": self.chat_window.theme,
                    "css_url": self._get_css_url(),
                    "js_url": self._get_js_url(),
                }
            )

        @self.app.post("/api/chat")
        async def post_message(request: Request):
            """Handle new message from user and return streaming response"""
            from fastapi.responses import StreamingResponse
            import json
            
            data = await request.json()
            content = data.get("content", "")
            
            def generate_response():
                """Stream response chunks as Server-Sent Events"""
                try:
                    # Call agent if registered
                    if self.chat_window.agent:
                        # Use synchronous streaming
                        for event in self.chat_window.agent.run_stream(content):
                            # Stream content chunks
                            if hasattr(event, 'content') and event.content:
                                yield f"data: {json.dumps({'chunk': event.content})}\n\n"
                        
                        # Send final done event
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    else:
                        # No agent registered
                        yield f"data: {json.dumps({'chunk': 'No agent registered. Please create an Agent with ui=chat_window'})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        
                except Exception as e:
                    # Send error
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
            
            return StreamingResponse(
                generate_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        # Serve IPC bridge
        @self.app.get("/bridge.js")
        async def serve_bridge():
            bridge_path = Path(__file__).parent / "templates" / "bridge.js"
            return FileResponse(str(bridge_path), media_type="application/javascript")

        # Serve static files if using local build
        if not cdn_url:
            ui_dir = Path(__file__).parent / "static"
            if ui_dir.exists():
                @self.app.get("/definable-chat.es.js")
                async def serve_js():
                    return FileResponse(str(ui_dir / "definable-chat.es.js"))

                @self.app.get("/chat-widget.css")
                async def serve_css():
                    return FileResponse(str(ui_dir / "chat-widget.css"))

    def _get_css_url(self) -> str:
        """Get CSS URL (CDN or local)."""
        return f"{self.cdn_url}/chat-widget.css" if self.cdn_url else "/chat-widget.css"
    
    def _get_js_url(self) -> str:
        """Get JS URL (CDN or local)."""
        return f"{self.cdn_url}/definable-chat.es.js" if self.cdn_url else "/definable-chat.es.js"

    def start(self):
        """Start the server (blocking call)."""
        import asyncio
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")
