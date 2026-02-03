"""FastAPI server for UI with Jinja2 templates and IPC bridge."""

from pathlib import Path
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates


class UIServer:
    """FastAPI server for chat UI with IPC bridge."""

    def __init__(self, chat_window, port: int = 8000, cdn_url: str = None):
        self.chat_window = chat_window
        self.port = port
        self.cdn_url = cdn_url
        self.app = FastAPI(title="Definable UI")
        
        templates_dir = Path(__file__).parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def serve_root(request: Request):
            return self.templates.TemplateResponse("chat.html", {
                "request": request,
                "title": self.chat_window.title,
                "theme": self.chat_window.theme,
                "css_url": self._get_css_url(),
                "js_url": self._get_js_url(),
            })

        @self.app.post("/api/chat")
        async def post_message(request: Request):
            data = await request.json()
            content = data.get("content", "")
            
            def generate():
                try:
                    if self.chat_window.agent:
                        for event in self.chat_window.agent.run_stream(content):
                            if hasattr(event, 'content') and event.content:
                                yield f"data: {json.dumps({'chunk': event.content})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    else:
                        yield f"data: {json.dumps({'chunk': 'No agent registered'})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            })

        @self.app.get("/bridge.js")
        async def serve_bridge():
            return FileResponse(Path(__file__).parent / "templates" / "bridge.js")

    def _get_css_url(self):
        return f"{self.cdn_url}/chat-widget.css" if self.cdn_url else "/chat-widget.css"
    
    def _get_js_url(self):
        return f"{self.cdn_url}/definable-chat.es.js" if self.cdn_url else "/definable-chat.es.js"

    def start(self):
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")
