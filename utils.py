from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime
from elevenlabs import ElevenLabs
from pathlib import Path
from gtts import gTTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)


class MCPOverloadedError(Exception):
    """Custom exception for MCP service overloads"""
    pass


def generate_valid_news_url(keyword: str) -> str:
    """
    Generate a Google News search URL for a keyword with optional sorting by latest
    
    Args:
        keyword: Search term to use in the news search
        
    Returns:
        str: Constructed Google News search URL
    """
    if not keyword or not keyword.strip():
        raise ValueError("Keyword cannot be empty")
    
    q = quote_plus(keyword.strip())
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def generate_news_urls_to_scrape(list_of_keywords):
    """
    Generate news URLs for multiple keywords
    
    Args:
        list_of_keywords: List of search terms
        
    Returns:
        dict: Dictionary mapping keywords to their news URLs
    """
    if not list_of_keywords:
        return {}
    
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        try:
            valid_urls_dict[keyword] = generate_valid_news_url(keyword)
        except ValueError as e:
            logger.warning(f"Skipping invalid keyword '{keyword}': {e}")
            continue
    
    return valid_urls_dict


def scrape_with_brightdata(url: str) -> str:
    """
    Scrape a URL using BrightData
    
    Args:
        url: URL to scrape
        
    Returns:
        str: Scraped content
        
    Raises:
        HTTPException: If scraping fails
    """
    api_key = os.getenv('BRIGHTDATA_API_KEY') or os.getenv('BRIGHTDATA_API_TOKEN')
    zone = os.getenv('BRIGHTDATA_WEB_UNLOCKER_ZONE') or os.getenv('WEB_UNLOCKER_ZONE')
    
    if not api_key:
        raise HTTPException(status_code=500, detail="BrightData API key not configured")
    
    if not zone:
        raise HTTPException(status_code=500, detail="BrightData zone not configured")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "zone": zone,
        "url": url,
        "format": "raw"
    }
    
    try:
        response = requests.post(
            "https://api.brightdata.com/request", 
            json=payload, 
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="BrightData request timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")


def clean_html_to_text(html_content: str) -> str:
    """
    Clean HTML content to plain text
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        str: Cleaned plain text
    """
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        return ""


def extract_headlines(cleaned_text: str) -> str:
    """
    Extract and concatenate headlines from cleaned news text content.
    
    Args:
        cleaned_text: Raw text from news page after HTML cleaning
        
    Returns:
        str: Combined headlines separated by newlines
    """
    if not cleaned_text:
        return ""
    
    headlines = []
    current_block = []
    
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    
    # Process lines to find headline blocks
    for line in lines:
        if line == "More":
            if current_block:
                # First line of block is headline
                headlines.append(current_block[0])
                current_block = []
        else:
            current_block.append(line)
    
    # Add any remaining block at end of text
    if current_block:
        headlines.append(current_block[0])
    
    return "\n".join(headlines)


def summarize_with_gemini_news_script(api_key: str, headlines: str) -> str:
    """
    Summarize multiple news headlines into a TTS-friendly broadcast news script using Gemini.
    
    Args:
        api_key: Gemini API key
        headlines: News headlines to summarize
        
    Returns:
        str: Generated news script
        
    Raises:
        HTTPException: If summarization fails
    """
    if not api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not provided")
    
    if not headlines or not headlines.strip():
        return "No news headlines available for summarization."
    
    system_prompt = """
You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.
The final output will be read aloud by a news anchor or text-to-speech engine. So:
- Do not include any special characters, emojis, formatting symbols, or markdown.
- Do not add any preamble or framing like "Here's your summary" or "Let me explain".
- Write in full, clear, spoken-language paragraphs.
- Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
- Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
- Start right away with the actual script, using transitions between topics if needed.
Remember: Your only output should be a clean script that is ready to be read out loud.
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=api_key,
            temperature=0.4,
            max_tokens=1000
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=headlines)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")


def generate_broadcast_news(api_key, news_data, reddit_data, topics):
    """
    Generate broadcast news using available sources
    
    Args:
        api_key: Gemini API key
        news_data: Dictionary containing news analysis
        reddit_data: Dictionary containing reddit analysis
        topics: List of topics to cover
        
    Returns:
        str: Generated broadcast script
        
    Raises:
        Exception: If generation fails
    """
    if not api_key:
        raise ValueError("Gemini API key is required")
    
    if not topics:
        return "No topics provided for broadcast generation."
    
    # Updated system message with flexible source handling
    system_prompt = """
    You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

    For each topic, STRUCTURE BASED ON AVAILABLE DATA:
    1. If news exists: "According to official reports..." + summary
    2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
    3. If both exist: Present news first, then Reddit reactions
    4. If neither exists: Skip the topic (shouldn't happen)

    Formatting rules:
    - ALWAYS start directly with the content, NO INTRODUCTIONS
    - Keep audio length 60-120 seconds per topic
    - Use natural speech transitions like "Meanwhile, online discussions..." 
    - Incorporate 1-2 short quotes from Reddit when available
    - Maintain neutral tone but highlight key sentiments
    - End with "To wrap up this segment..." summary

    Write in full paragraphs optimized for speech synthesis. Avoid markdown.
    """

    try:
        topic_blocks = []
        for topic in topics:
            news_content = news_data.get("news_analysis", {}).get(topic, '') if news_data else ''
            reddit_content = reddit_data.get("reddit_analysis", {}).get(topic, '') if reddit_data else ''
            
            context = []
            if news_content:
                context.append(f"OFFICIAL NEWS CONTENT:\n{news_content}")
            if reddit_content:
                context.append(f"REDDIT DISCUSSION CONTENT:\n{reddit_content}")
            
            if context:  # Only include topics with actual content
                topic_blocks.append(
                    f"TOPIC: {topic}\n\n" +
                    "\n\n".join(context)
                )

        if not topic_blocks:
            return "No content available for any of the specified topics."

        user_prompt = (
            "Create broadcast segments for these topics using available sources:\n\n" +
            "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=4000
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating broadcast news: {e}")
        raise e


def text_to_audio_elevenlabs_sdk(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    output_dir: str = "audio",
    api_key: str = None
) -> str:
    """
    Converts text to speech using ElevenLabs SDK and saves it to audio/ directory.

    Args:
        text: Text to convert to speech
        voice_id: ElevenLabs voice ID
        model_id: ElevenLabs model ID
        output_format: Output audio format
        output_dir: Directory to save audio files
        api_key: ElevenLabs API key

    Returns:
        str: Path to the saved audio file.
        
    Raises:
        ValueError: If API key is missing or text is empty
        Exception: If TTS conversion fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    api_key = api_key or os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key is required.")

    try:
        # Initialize client
        client = ElevenLabs(api_key=api_key)

        # Get the audio generator
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename
        filename = f"tts_elevenlabs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)

        # Write audio chunks to file
        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        logger.info(f"Audio saved to: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {e}")
        raise e


def tts_to_audio(text: str, language: str = 'en') -> str:
    """
    Convert text to speech using gTTS (Google Text-to-Speech) and save to file.
    
    Args:
        text: Input text to convert
        language: Language code (default: 'en')
    
    Returns:
        str: Path to saved audio file, or None if failed
    
    Example:
        tts_to_audio("Hello world", "en")
    """
    if not text or not text.strip():
        logger.error("Text cannot be empty")
        return None
    
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AUDIO_DIR / f"tts_gtts_{timestamp}.mp3"
        
        # Create TTS object and save
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(filename))
        
        logger.info(f"Audio saved to: {filename}")
        return str(filename)
    except Exception as e:
        logger.error(f"gTTS Error: {str(e)}")
        return None


def get_api_keys():
    """
    Get API keys from environment variables
    
    Returns:
        dict: Dictionary containing API keys
    """
    return {
        'brightdata_api_key': os.getenv('BRIGHTDATA_API_KEY') or os.getenv('BRIGHTDATA_API_TOKEN'),
        'brightdata_zone': os.getenv('BRIGHTDATA_WEB_UNLOCKER_ZONE') or os.getenv('WEB_UNLOCKER_ZONE'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),
        'eleven_api_key': os.getenv('ELEVEN_API_KEY'),
    }


def validate_environment():
    """
    Validate that required environment variables are set
    
    Raises:
        ValueError: If required environment variables are missing
    """
    api_keys = get_api_keys()
    missing_keys = []
    
    if not api_keys['brightdata_api_key']:
        missing_keys.append('BRIGHTDATA_API_KEY or BRIGHTDATA_API_TOKEN')
    if not api_keys['brightdata_zone']:
        missing_keys.append('BRIGHTDATA_WEB_UNLOCKER_ZONE or WEB_UNLOCKER_ZONE')
    if not api_keys['gemini_api_key']:
        missing_keys.append('GEMINI_API_KEY')
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Validate environment
        validate_environment()
        logger.info("Environment validation passed")
        
        # Example usage
        keywords = ["artificial intelligence", "climate change"]
        urls = generate_news_urls_to_scrape(keywords)
        
        for keyword, url in urls.items():
            logger.info(f"Generated URL for '{keyword}': {url}")
            
    except Exception as e:
        logger.error(f"Error: {e}")