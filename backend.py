from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import logging
import uvicorn

from models import NewsRequest
from utils import generate_broadcast_news, text_to_audio_elevenlabs_sdk
from news_scraper import NewsScraper
from reddit__scraper import scrape_reddit_topics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Audio Generator API",
    description="Generate audio summaries from news and Reddit content",
    version="1.0.0"
)

# Load environment variables
load_dotenv()

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure audio directory exists
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Validate required environment variables
required_env_vars = ["GEMINI_API_KEY", "ELEVEN_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {missing_vars}")
    raise ValueError(f"Missing required environment variables: {missing_vars}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "News Audio Generator API is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "audio_dir_exists": AUDIO_DIR.exists(),
        "env_vars_loaded": {
            "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
            "ELEVEN_API_KEY": bool(os.getenv("ELEVEN_API_KEY")),
            "API_TOKEN": bool(os.getenv("API_TOKEN")),
        }
    }


@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest, audio: bool = Query(False)):
    """
    Generate news summary (and optionally audio) from news and/or Reddit content
    """
    try:
        logger.info(f"Processing request for topics: {request.topics}, source: {request.source_type}")
        
        # Validate request
        if not request.topics:
            raise HTTPException(status_code=400, detail="Topics list cannot be empty")
        
        if request.source_type not in ["news", "reddit", "both"]:
            raise HTTPException(
                status_code=400, 
                detail="source_type must be 'news', 'reddit', or 'both'"
            )
        
        results = {}
        
        # Scrape news data
        if request.source_type in ["news", "both"]:
            try:
                news_scraper = NewsScraper()
                news_results = await news_scraper.scrape_news(request.topics)
                if news_results:
                    results["news"] = news_results.get("news_analysis", {})
                logger.info(f"News data scraped successfully: {len(results.get('news', {}))} items")
            except Exception as e:
                logger.error(f"Error scraping news: {str(e)}")
                results["news"] = {}
        
        # Scrape Reddit data
        if request.source_type in ["reddit", "both"]:
            try:
                reddit_results = await scrape_reddit_topics(request.topics)
                if reddit_results:
                    results["reddit"] = reddit_results.get("reddit_analysis", {})
                logger.info(f"Reddit data scraped successfully: {len(results.get('reddit', {}))} items")
            except Exception as e:
                logger.error(f"Error scraping Reddit: {str(e)}")
                results["reddit"] = {}
        
        # Check if we have any data
        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})
        
        if not news_data and not reddit_data:
            raise HTTPException(
                status_code=404, 
                detail="No data found for the specified topics and sources"
            )
        
        # Generate news summary
        logger.info("Generating news summary...")
        try:
            news_summary = generate_broadcast_news(
                api_key=os.getenv("GEMINI_API_KEY"),
                news_data=news_data,
                reddit_data=reddit_data,
                topics=request.topics
            )
            
            if not news_summary or len(news_summary.strip()) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate news summary - empty response"
                )
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate news summary: {str(e)}"
            )
        
        # If only text is requested, return it
        if not audio:
            return {"summary": news_summary}
        
        # Generate audio
        logger.info("Converting text to audio...")
        try:
            audio_path = text_to_audio_elevenlabs_sdk(
                text=news_summary,
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                output_dir="audio"
            )
            
            if not audio_path:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate audio file"
                )
            
            audio_file_path = Path(audio_path)
            if not audio_file_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Generated audio file not found"
                )
            
            # Read and return audio file
            try:
                audio_data = audio_file_path.read_bytes()
                return Response(
                    content=audio_data,
                    media_type="audio/mpeg",
                    headers={"Content-Disposition": f"attachment; filename={audio_file_path.name}"}
                )
                
            except IOError as e:
                logger.error(f"Error reading audio file: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error reading audio file: {str(e)}"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate audio: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_news_audio: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    # Validate environment before starting server
    try:
        # Test if we can access required services
        logger.info("Starting News Audio Generator API...")
        logger.info(f"Audio directory: {AUDIO_DIR.absolute()}")
        
        uvicorn.run(
            "backend:app",
            host="0.0.0.0",
            port=1234,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise