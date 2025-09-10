"""
FastAPI application for Catan Assistant.
Provides REST API endpoints for the LLM-powered Catan analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging

from ..orchestration.langgraph_pipeline import CatanOrchestrator, StreamingOrchestrator
from ..orchestration.langchain_tools import CatanToolkit
from ..core.game_state import GameState
from ..utils.config import load_config
from ..utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Global orchestrator instances
orchestrator: Optional[CatanOrchestrator] = None
streaming_orchestrator: Optional[StreamingOrchestrator] = None
toolkit: Optional[CatanToolkit] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator, streaming_orchestrator, toolkit
    
    # Startup
    logger.info("Starting Catan Assistant API...")
    config = load_config()
    
    orchestrator = CatanOrchestrator(config)
    streaming_orchestrator = StreamingOrchestrator(config)
    toolkit = CatanToolkit(config)
    
    logger.info("✅ Catan Assistant API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Catan Assistant API...")

# Create FastAPI app
app = FastAPI(
    title="Catan Assistant API",
    description="LLM-powered Settlers of Catan game analysis and strategy assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for game analysis."""
    game_state: Optional[Dict[str, Any]] = None
    capture_screen: bool = True
    player_perspective: str = "Blue"
    include_detailed_analysis: bool = True

class AnalysisResponse(BaseModel):
    """Response model for game analysis."""
    success: bool
    recommendation: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    game_context: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int = 0

class StreamingResponse(BaseModel):
    """Response model for streaming analysis."""
    step: str
    status: str
    data: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: Dict[str, str]

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Catan Assistant API",
        "version": "1.0.0",
        "description": "LLM-powered Settlers of Catan analysis",
        "endpoints": {
            "analyze": "/analyze",
            "stream": "/analyze/stream",
            "tools": "/tools",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "orchestrator": "running" if orchestrator else "not_initialized",
            "streaming": "running" if streaming_orchestrator else "not_initialized",
            "toolkit": "running" if toolkit else "not_initialized"
        }
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_game(request: AnalysisRequest):
    """
    Analyze current Catan game state and provide strategic recommendations.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # If game state is provided, use it directly
        if request.game_state:
            # Process provided game state
            game_state = GameState.from_json(request.game_state)
            decision_result = orchestrator.decision_module.forward(game_state)
            
            result = {
                "recommendation": {
                    "action": decision_result["best_action"],
                    "rationale": decision_result["rationale"],
                    "confidence": decision_result["confidence"]
                },
                "analysis": {
                    "position": decision_result["position_analysis"],
                    "priorities": decision_result["strategic_priorities"],
                    "threats": decision_result["opponent_threats"]
                } if request.include_detailed_analysis else None,
                "game_context": {
                    "turn": game_state.turn,
                    "player": game_state.current_player,
                    "victory_points": game_state.get_current_player_state().victory_points
                }
            }
        else:
            # Use full pipeline with screen capture
            result = await orchestrator.process_game()
            if not result["success"]:
                raise HTTPException(status_code=400, detail=result["error"])
            result = result["result"]
        
        end_time = asyncio.get_event_loop().time()
        processing_time = int((end_time - start_time) * 1000)
        
        return AnalysisResponse(
            success=True,
            recommendation=result["recommendation"],
            analysis=result.get("analysis"),
            game_context=result.get("game_context"),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        end_time = asyncio.get_event_loop().time()
        processing_time = int((end_time - start_time) * 1000)
        
        return AnalysisResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time
        )

@app.get("/analyze/stream")
async def stream_analysis():
    """
    Stream the game analysis process in real-time.
    """
    if not streaming_orchestrator:
        raise HTTPException(status_code=503, detail="Streaming orchestrator not initialized")
    
    async def generate_stream():
        try:
            async for step_update in streaming_orchestrator.stream_analysis():
                yield f"data: {step_update}\n\n"
        except Exception as e:
            yield f"data: {{'error': '{str(e)}'}}\n\n"
    
    from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
    return FastAPIStreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/tools", response_model=Dict[str, Any])
async def get_available_tools():
    """Get information about available tools."""
    if not toolkit:
        raise HTTPException(status_code=503, detail="Toolkit not initialized")
    
    return {
        "tools": toolkit.get_tool_names(),
        "descriptions": toolkit.get_tool_descriptions(),
        "count": len(toolkit.get_tools())
    }

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, parameters: Dict[str, Any]):
    """Execute a specific tool with given parameters."""
    if not toolkit:
        raise HTTPException(status_code=503, detail="Toolkit not initialized")
    
    tools = {tool.name: tool for tool in toolkit.get_tools()}
    
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    try:
        tool = tools[tool_name]
        result = tool.run(parameters)
        return {
            "success": True,
            "tool": tool_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e)
        }

@app.post("/validate-game-state")
async def validate_game_state(game_state: Dict[str, Any]):
    """Validate a game state JSON structure."""
    try:
        validated_state = GameState.from_json(game_state)
        return {
            "valid": True,
            "player_count": len(validated_state.players),
            "current_player": validated_state.current_player,
            "turn": validated_state.turn,
            "legal_actions_count": len(validated_state.legal_actions)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "catan_assistant.api.fastapi_app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()