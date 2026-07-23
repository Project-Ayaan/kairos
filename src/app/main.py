import secrets

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes
)

from app.core.config import settings
from app.core.rag import KairosRAGPipeline
from app.a2a.executor import KairosAgentExecutor
from app.a2a.agent_card import get_agent_card

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Kairos CDSS A2A Service",
    version="0.1.0",
    description="Evidence-grounded Clinical Decision Support System A2A service."
)

# 2. Add Conditional A2A Auth Middleware
class A2AAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Authenticate A2A POST execution requests on the /a2a endpoint
        if request.url.path == "/a2a" and request.method == "POST":
            if settings.a2a_api_key:
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid Authorization header."}
                    )
                
                token = auth_header.split(" ", 1)[1]
                if not secrets.compare_digest(token, settings.a2a_api_key):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid A2A API key."}
                    )
                    
        return await call_next(request)

app.add_middleware(A2AAuthMiddleware)

# 3. Setup RAG pipeline, task store, and A2A executor
pipeline = KairosRAGPipeline(
    qdrant_url=settings.qdrant_url, 
    collection_name=settings.collection_name
)
executor = KairosAgentExecutor(pipeline=pipeline)
task_store = InMemoryTaskStore()
agent_card = get_agent_card()

# 4. Instantiate the A2A request handler
request_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=task_store,
    agent_card=agent_card
)

# 5. Build and register A2A routes
agent_card_routes = create_agent_card_routes(
    agent_card=agent_card,
    card_url='/.well-known/agent.json'
)
jsonrpc_routes = create_jsonrpc_routes(
    request_handler=request_handler,
    rpc_url='/a2a'
)

add_a2a_routes_to_fastapi(
    app,
    agent_card_routes=agent_card_routes,
    jsonrpc_routes=jsonrpc_routes
)

# 6. Basic health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}
