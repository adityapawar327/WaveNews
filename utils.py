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

class BrightDataError(Exception):
    """Custom exception for BrightData API errors"""
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
        raise ValueError("Keywords list cannot be empty")
    
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        try:
            url = generate_valid_news_url(keyword)
            valid_urls_dict[keyword] = url
        except ValueError:
            logger.warning(f"Skipping invalid keyword: {keyword}")
    
    return valid_urls_dict


def scrape_with_brightdata(url: str) -> str:
    """
    Scrape a URL using BrightData Web Scraper (latest API pattern)
    """
    api_key = os.getenv('BRIGHTDATA_API_KEY') or os.getenv('API_TOKEN')
    zone = os.getenv('BRIGHTDATA_WEB_UNLOCKER_ZONE') or os.getenv('WEB_UNLOCKER_ZONE')
    if not api_key or not zone:
        raise HTTPException(status_code=500, detail="BrightData API key or zone missing in environment variables")
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
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error: {str(e)}")


def clean_html_to_text(html_content: str) -> str:
    """Clean HTML content to plain text"""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()


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
            model="gemini-2.5-flash",
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
            model="gemini-2.5-flash",
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
    Returns:
        str: Path to the saved audio file.
    """
    try:
        api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key is required.")
        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )
        os.makedirs(output_dir, exist_ok=True)
        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return filepath
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ElevenLabs error: {str(e)}")

def tts_to_audio(text: str, language: str = 'en') -> str:
    """
    Convert text to speech using gTTS (Google Text-to-Speech) and save to file.
    Returns: str: Path to saved audio file
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AUDIO_DIR / f"tts_{timestamp}.mp3"
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(filename))
        return str(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS error: {str(e)}")

def get_api_keys():
    """Get and validate API keys"""
    keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "ELEVEN_API_KEY": os.getenv("ELEVEN_API_KEY"),
        "BRIGHTDATA_API_KEY": os.getenv("BRIGHTDATA_API_KEY"),
        "API_TOKEN": os.getenv("API_TOKEN")
    }
    
    missing = [k for k, v in keys.items() if not v]
    if missing:
        raise ValueError(f"Missing required API keys: {', '.join(missing)}")
    
    return keys

def validate_environment():
    """Validate environment setup and test API connections"""
    errors = []
    warnings = []
    
    # Check API Keys
    required_keys = {
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "ELEVEN_API_KEY": os.getenv("ELEVEN_API_KEY"),
        "API_TOKEN": os.getenv("API_TOKEN"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE")
    }
    
    missing_keys = [k for k, v in required_keys.items() if not v]
    if missing_keys:
        errors.append(f"Missing required API keys: {', '.join(missing_keys)}")
    
    # Check audio directory
    try:
        if not AUDIO_DIR.exists():
            AUDIO_DIR.mkdir(parents=True)
            logger.info(f"Created audio directory at {AUDIO_DIR}")
    except Exception as e:
        errors.append(f"Failed to create audio directory: {str(e)}")
    
    # Test BrightData connection if credentials are present
    if all(k not in missing_keys for k in ["API_TOKEN", "WEB_UNLOCKER_ZONE"]):
        try:
            # Simple test URL that should always work
            test_url = "http://example.com"
            logger.info("Testing BrightData connection...")
            content = scrape_with_brightdata(test_url)
            if content and len(content) > 0:
                logger.info("✓ BrightData connection test successful")
            else:
                errors.append("BrightData returned empty response")
        except Exception as e:
            errors.append(f"BrightData connection test failed: {str(e)}")
    else:
        warnings.append("Skipping BrightData test - missing credentials")
    
    # Test ElevenLabs connection
    if "ELEVEN_API_KEY" not in missing_keys:
        try:
            eleven = ElevenLabs(api_key=required_keys["ELEVEN_API_KEY"])
            # Get available voices to test connection
            voices = eleven.voices.get_all()
            if not voices:
                warnings.append("ElevenLabs: No voices available")
            else:
                logger.info("✓ ElevenLabs connection test successful")
        except Exception as e:
            errors.append(f"ElevenLabs connection test failed: {str(e)}")
    else:
        warnings.append("Skipping ElevenLabs test - missing API key")
    
    # Test Gemini connection
    if "GEMINI_API_KEY" not in missing_keys:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=required_keys["GEMINI_API_KEY"]
            )
            test_response = llm.invoke("Test connection")
            if test_response:
                logger.info("✓ Gemini connection test successful")
        except Exception as e:
            errors.append(f"Gemini connection test failed: {str(e)}")
    else:
        warnings.append("Skipping Gemini test - missing API key")
    
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Environment validation failed:\n{error_msg}")
        raise ValueError(error_msg)
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info("✓ Environment validation completed successfully")
    return True


# Example usage and testing
if __name__ == "__main__":
    try:
        # Step 1: Validate environment and connections
        validate_environment()
        
        # Step 2: Test news URL generation
        logger.info("\n=== Testing News URL Generation ===")
        keywords = ["artificial intelligence", "climate change"]
        urls = generate_news_urls_to_scrape(keywords)
        for keyword, url in urls.items():
            logger.info(f"Generated URL for '{keyword}': {url}")
        
        # Step 3: Test BrightData scraping
        logger.info("\n=== Testing BrightData Scraping ===")
        for keyword, url in urls.items():
            try:
                logger.info(f"Scraping content for '{keyword}'...")
                content = scrape_with_brightdata(url)
                cleaned_text = clean_html_to_text(content)
                headlines = extract_headlines(cleaned_text)
                logger.info(f"Found {len(headlines.split('\\n'))} headlines for '{keyword}'")
            except Exception as e:
                logger.error(f"Error scraping '{keyword}': {str(e)}")
        
        # Step 4: Test Gemini summarization
        logger.info("\n=== Testing Gemini Summarization ===")
        try:
            test_headlines = """
            Latest developments in AI show promising results
            Climate change impacts worse than expected
            New AI models break performance records
            Global temperature rise accelerates
            """
            summary = summarize_with_gemini_news_script(
                os.getenv("GEMINI_API_KEY"),
                test_headlines
            )
            logger.info("Successfully generated summary")
            logger.info(f"Summary preview: {summary[:200]}...")
        except Exception as e:
            logger.error(f"Error in Gemini summarization: {str(e)}")
        
        # Step 5: Test text-to-speech
        logger.info("\n=== Testing Text-to-Speech ===")
        try:
            test_text = "This is a test of the text-to-speech system."
            # Test ElevenLabs
            audio_path = text_to_audio_elevenlabs_sdk(
                text=test_text,
                output_dir="audio"
            )
            logger.info(f"Successfully generated audio file: {audio_path}")
            
            # Test fallback TTS
            fallback_path = tts_to_audio(test_text)
            logger.info(f"Successfully generated fallback audio: {fallback_path}")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
        
        logger.info("\n=== Test Complete ===")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise