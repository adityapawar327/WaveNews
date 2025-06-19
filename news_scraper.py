import asyncio
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from aiolimiter import AsyncLimiter
from tenacity import (
    retry, 
    retry_if_exception_type, 
    stop_after_attempt, 
    wait_exponential,
    RetryError
)
from dotenv import load_dotenv
from utils import (
    generate_news_urls_to_scrape,
    scrape_with_brightdata,
    clean_html_to_text,
    extract_headlines,
    summarize_with_gemini_news_script
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class NewsScrapingError(Exception):
    """Custom exception for news scraping errors"""
    pass

class BrightDataError(Exception):
    """Custom exception for BrightData API errors"""
    pass

class NewsScraper:
    def __init__(self):
        """Initialize the news scraper with rate limiting and configuration"""
        # Rate limiter: 3 requests per second (more conservative)
        self._rate_limiter = AsyncLimiter(3, 1)
        
        # Validate required environment variables
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.brightdata_token = os.getenv("API_TOKEN")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        if not self.brightdata_token:
            logger.warning("API_TOKEN not found - some scraping features may not work")
        
        logger.info("NewsScraper initialized successfully")

    def _validate_topics(self, topics: List[str]) -> List[str]:
        """Validate and clean topics list"""
        if not topics:
            raise ValueError("Topics list cannot be empty")
        
        # Filter out empty or invalid topics
        valid_topics = [topic.strip() for topic in topics if topic and topic.strip()]
        
        if not valid_topics:
            raise ValueError("No valid topics provided")
        
        # Limit number of topics to prevent overload
        if len(valid_topics) > 10:
            logger.warning(f"Too many topics ({len(valid_topics)}), limiting to first 10")
            valid_topics = valid_topics[:10]
        
        return valid_topics

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=30),
        retry=retry_if_exception_type((BrightDataError, ConnectionError, TimeoutError))
    )
    async def _scrape_topic(self, topic: str) -> str:
        """Scrape news for a single topic with retry logic"""
        logger.info(f"Scraping news for topic: {topic}")
        
        try:
            # Rate limiting
            async with self._rate_limiter:
                # Step 1: Generate URLs
                logger.debug(f"Generating URLs for topic: {topic}")
                urls_dict = generate_news_urls_to_scrape([topic])
                
                if not urls_dict or topic not in urls_dict:
                    raise NewsScrapingError(f"Failed to generate URLs for topic: {topic}")
                
                urls = urls_dict[topic]
                if not urls:
                    raise NewsScrapingError(f"No URLs generated for topic: {topic}")
                
                logger.debug(f"Generated {len(urls) if isinstance(urls, list) else 1} URLs for topic: {topic}")
                
                # Step 2: Scrape with BrightData
                logger.debug(f"Scraping content for topic: {topic}")
                try:
                    search_html = await self._async_scrape_with_brightdata(urls)
                except Exception as e:
                    logger.error(f"BrightData scraping failed for topic '{topic}': {str(e)}")
                    raise BrightDataError(f"Failed to scrape content for topic '{topic}': {str(e)}")
                
                if not search_html or search_html.strip() == "":
                    raise NewsScrapingError(f"Empty content returned for topic: {topic}")
                
                # Step 3: Clean HTML
                logger.debug(f"Cleaning HTML content for topic: {topic}")
                try:
                    clean_text = clean_html_to_text(search_html)
                except Exception as e:
                    logger.error(f"HTML cleaning failed for topic '{topic}': {str(e)}")
                    raise NewsScrapingError(f"Failed to clean HTML for topic '{topic}': {str(e)}")
                
                if not clean_text or clean_text.strip() == "":
                    raise NewsScrapingError(f"No clean text extracted for topic: {topic}")
                
                # Step 4: Extract headlines
                logger.debug(f"Extracting headlines for topic: {topic}")
                try:
                    headlines = extract_headlines(clean_text)
                except Exception as e:
                    logger.error(f"Headline extraction failed for topic '{topic}': {str(e)}")
                    raise NewsScrapingError(f"Failed to extract headlines for topic '{topic}': {str(e)}")
                
                if not headlines:
                    logger.warning(f"No headlines extracted for topic: {topic}")
                    headlines = [f"No specific headlines found for {topic}"]
                
                # Step 5: Summarize with Gemini
                logger.debug(f"Generating summary for topic: {topic}")
                try:
                    summary = await self._async_summarize_with_gemini(headlines, topic)
                except Exception as e:
                    logger.error(f"Gemini summarization failed for topic '{topic}': {str(e)}")
                    raise NewsScrapingError(f"Failed to generate summary for topic '{topic}': {str(e)}")
                
                if not summary or summary.strip() == "":
                    raise NewsScrapingError(f"Empty summary generated for topic: {topic}")
                
                logger.info(f"Successfully processed topic: {topic}")
                return summary
                
        except (BrightDataError, ConnectionError, TimeoutError):
            # These exceptions will be retried
            raise
        except Exception as e:
            # Other exceptions won't be retried
            logger.error(f"Non-retryable error for topic '{topic}': {str(e)}")
            raise NewsScrapingError(f"Failed to process topic '{topic}': {str(e)}")

    async def _async_scrape_with_brightdata(self, urls) -> str:
        """Async wrapper for BrightData scraping"""
        try:
            # Run the synchronous scraping function in a thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, scrape_with_brightdata, urls),
                timeout=60  # 60 second timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError("BrightData scraping timeout")
        except Exception as e:
            raise BrightDataError(f"BrightData scraping error: {str(e)}")

    async def _async_summarize_with_gemini(self, headlines: List[str], topic: str) -> str:
        """Async wrapper for Gemini summarization"""
        try:
            # Run the synchronous summarization function in a thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    summarize_with_gemini_news_script,
                    self.api_key,
                    headlines,
                    topic  # Pass topic for better context
                ),
                timeout=30  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError("Gemini summarization timeout")
        except Exception as e:
            raise Exception(f"Gemini summarization error: {str(e)}")

    async def scrape_news(self, topics: List[str]) -> Dict[str, Any]:
        """
        Scrape and analyze news articles for multiple topics
        
        Args:
            topics: List of topics to scrape news for
            
        Returns:
            Dictionary containing news analysis results and metadata
        """
        if not topics:
            logger.warning("No topics provided for news scraping")
            return {
                "news_analysis": {},
                "total_topics": 0,
                "successful_topics": 0,
                "failed_topics": 0,
                "errors": ["No topics provided"],
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Validate topics
            valid_topics = self._validate_topics(topics)
            logger.info(f"Starting news scraping for {len(valid_topics)} topics: {valid_topics}")
            
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Topic validation failed: {error_msg}")
            return {
                "news_analysis": {},
                "total_topics": len(topics),
                "successful_topics": 0,
                "failed_topics": len(topics),
                "errors": [error_msg],
                "timestamp": datetime.now().isoformat()
            }
        
        results = {}
        errors = []
        
        # Process topics with controlled concurrency
        semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent topics
        
        async def process_topic_with_semaphore(topic: str):
            async with semaphore:
                try:
                    result = await self._scrape_topic(topic)
                    return topic, result, None
                except RetryError as e:
                    error_msg = f"All retry attempts failed for topic '{topic}': {str(e)}"
                    logger.error(error_msg)
                    return topic, None, error_msg
                except Exception as e:
                    error_msg = f"Unexpected error for topic '{topic}': {str(e)}"
                    logger.error(error_msg)
                    return topic, None, error_msg
        
        # Create tasks for all topics
        tasks = [process_topic_with_semaphore(topic) for topic in valid_topics]
        
        # Process all topics
        try:
            # Add overall timeout for all topics
            completed_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute total timeout
            )
            
            for result in completed_results:
                if isinstance(result, Exception):
                    error_msg = f"Task failed with exception: {str(result)}"
                    errors.append(error_msg)
                    continue
                
                topic, summary, error = result
                if error:
                    errors.append(error)
                    results[topic] = f"Error: {error}"
                else:
                    results[topic] = summary
                
        except asyncio.TimeoutError:
            error_msg = "Overall scraping timeout exceeded"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        # Add small delay between topics to be respectful
        await asyncio.sleep(1)
        
        # Prepare final results
        successful_count = len([v for v in results.values() if not str(v).startswith("Error:")])
        failed_count = len(valid_topics) - successful_count
        
        final_result = {
            "news_analysis": results,
            "total_topics": len(valid_topics),
            "successful_topics": successful_count,
            "failed_topics": failed_count,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "rate_limit": "3 requests/second",
            "timeout_settings": {
                "brightdata_timeout": "60s",
                "gemini_timeout": "30s",
                "overall_timeout": "300s"
            }
        }
        
        logger.info(f"News scraping completed. Success: {successful_count}/{len(valid_topics)}")
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during scraping")
        
        return final_result

# Utility functions for testing and debugging
async def test_news_scraping():
    """Test function for debugging"""
    scraper = NewsScraper()
    test_topics = ["artificial intelligence", "climate change", "space exploration"]
    
    logger.info("Starting news scraping test...")
    result = await scraper.scrape_news(test_topics)
    
    print("\n=== Test Results ===")
    print(f"Total topics: {result['total_topics']}")
    print(f"Successful: {result['successful_topics']}")
    print(f"Failed: {result['failed_topics']}")
    
    if result['errors']:
        print(f"\nErrors encountered:")
        for error in result['errors']:
            print(f"  - {error}")
    
    print(f"\nNews Analysis Results:")
    for topic, analysis in result['news_analysis'].items():
        print(f"\n--- {topic.upper()} ---")
        if str(analysis).startswith("Error:"):
            print(f"❌ {analysis}")
        else:
            print(f"✅ {analysis[:300]}{'...' if len(str(analysis)) > 300 else ''}")

def create_news_scraper() -> NewsScraper:
    """Factory function to create a properly configured news scraper"""
    return NewsScraper()

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_news_scraping())