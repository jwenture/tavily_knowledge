from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from urllib.parse import unquote
import os
import asyncio
from pathlib import Path
from pydantic import BaseModel
from app.process import ProcessFlow
from app.log import logger, lifespan
from dotenv import load_dotenv
import os
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

USERNAME = os.getenv("API_USERNAME", "tavilytester")
PASSWORD = os.getenv("API_PASSWORD", "tavilytesting")

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


class SearchQuery(BaseModel):
    query: str

app = FastAPI(lifespan=lifespan)
security = HTTPBasic()
pflow = ProcessFlow(tavily_api_key=tavily_api_key, openai_api_key=openai_api_key)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/lib", StaticFiles(directory="lib"), name="lib")

templates = Jinja2Templates(directory="templates")

def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Configuration
OUTPUT_HTML_PATH = "static/output.html"
OUTPUT_PNG_PATH = "static/graph.png"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Set this in your environment


@app.get("/test-error")
async def test_error(username: str = Depends(authenticate_user)):
    try:
        1 / 0
    except Exception as e:
        logger.error("Division by zero error occurred")
        return {"error": str(e)}

@app.get('/get-node/{node_id}')
def get_node(node_id: str, username: str = Depends(authenticate_user)):
    """Return full node data by ID (node_id is a string)"""
    decoded_node_id = unquote(node_id)
    decoded_node_id = decoded_node_id.strip().lower().replace(" ", "_")
    logger.info(f"Requested node_id: {decoded_node_id}")
    logger.info(f"Graph nodes: {pflow.graph_generator.graph.nodes}")
    if decoded_node_id not in pflow.graph_generator.graph.nodes:
        raise HTTPException(status_code=404, detail="Item not found")
    node = pflow.graph_generator.graph.nodes.get(decoded_node_id, {})
    if node:
        return node
    logger.warning(f"Error in get_node for node_id={node_id}")
    raise HTTPException(status_code=404, detail="Node not found")

@app.get('/redraw')
def redraw_graph( username: str = Depends(authenticate_user)):
    pflow.graph_generator.load_graph()
    pflow.graph_generator.create_html_graph()
    return {"success": True}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, username: str = Depends(authenticate_user)):
    """Serve the main page with current graph (if exists)"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "has_graph": os.path.exists(OUTPUT_HTML_PATH)}
    )

@app.post("/search")
async def search_papers(search_query: SearchQuery, username: str = Depends(authenticate_user)):
    """Handle search requests and generate knowledge graph"""
    query = search_query.query
    logger.info(f"Received query: {query}")

    try:
        search_results = pflow.search_and_process(
            query=query,
        )
        
        return search_results
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/graph")
async def get_graph(username: str = Depends(authenticate_user)):
    """Serve the latest graph HTML"""
    if os.path.exists(OUTPUT_HTML_PATH):
        return FileResponse(OUTPUT_HTML_PATH)
    return HTMLResponse("<p>No graph generated yet. Submit a search query first.</p>")


@app.delete('/delete-node/{node_id}')
def delete_node(node_id: str, username: str = Depends(authenticate_user)):
    """Delete node by ID (string) and all connected edges"""
    decoded_node_id = unquote(node_id)
    decoded_node_id = decoded_node_id.strip().lower().replace(" ", "_")
    logger.info(f"Requested node_id: {decoded_node_id}")
    # Check existence
    #if decoded_node_id not in pflow.graph_generator.graph.nodes:
    #    raise HTTPException(status_code=404, detail="Node not found")

    # Remove node
    pflow.graph_generator.graph.remove_node(decoded_node_id)
    pflow.graph_generator.save_graph()
    
    pflow.graph_generator.create_html_graph()

    return {"success": True}

@app.on_event("startup")
async def startup_event():
    for route in app.routes:
        print(f"Route: {route.path}")


if __name__ == "__main__":
    
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#