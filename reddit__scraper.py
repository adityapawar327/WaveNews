from typing import List, Dict, Any
import os
import asyncio
import logging
from datetime import datetime, timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
from tenacity import (
    retry, 
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from aiolimiter import AsyncLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Calculate date range for filtering
two_weeks_ago = datetime.today() - timedelta(days=14) 
two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')

class MCPOverloadedError(Exception):
    """Custom exception for MCP service overload"""
    pass

class RedditScrapingError(Exception):
    """Custom exception for Reddit scraping errors"""
    pass

# Rate limiter: 1 request per 15 seconds
mcp_limiter = AsyncLimiter(1, 15)

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")
WEB_UNLOCKER_ZONE = os.getenv("WEB_UNLOCKER_ZONE")

# Validate required environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": API_TOKEN,
        "WEB_UNLOCKER_ZONE": WEB_UNLOCKER_ZONE,
    },
    args=["@brightdata/mcp"],
)

def gemini_summarize_reddit(topic: str, reddit_content: str) -> str:
    """
    Summarize Reddit content using Gemini AI
    """
    if not reddit_content or reddit_content.strip() == "":
        logger.warning(f"Empty Reddit content for topic: {topic}")
        return f"No Reddit content found for topic '{topic}'"
    
    system_prompt = f"""
You are a Reddit analysis expert. Analyze the provided Reddit content about '{topic}' and create a comprehensive summary.

Requirements:
- Focus only on posts from after {two_weeks_ago_str}
- Provide clear, factual analysis
- Maintain objectivity and avoid bias
- Quote interesting comments without mentioning usernames
"""

    user_prompt = f"""
Analyze the following Reddit posts about '{topic}' and provide a comprehensive summary including:

1. Main discussion points and themes
2. Key opinions and perspectives expressed
3. Notable trends or patterns in the discussions
4. Interesting quotes from comments (anonymized)
5. Overall sentiment analysis (positive/neutral/negative)
6. Community engagement level and discussion quality

Reddit Content:
{reddit_content[:8000]}  # Limit content to avoid token limits
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Updated to more recent model
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,  # Slightly lower for more consistent results
            max_tokens=1500,  # Increased for more detailed summaries
            timeout=30  # Add timeout
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if not response or not response.content:
            logger.error(f"Empty response from Gemini for topic: {topic}")
            return f"Failed to generate summary for topic '{topic}' - empty response"
        
        logger.info(f"Successfully generated summary for topic: {topic}")
        return response.content
        
    except Exception as e:
        logger.error(f"Gemini API error for topic '{topic}': {str(e)}")
        return f"Error generating summary for topic '{topic}': {str(e)}"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=15, max=120),  # Increased backoff
    retry=retry_if_exception_type((MCPOverloadedError, ConnectionError)),
    reraise=True
)
async def process_topic(agent, topic: str) -> str:
    """
    Process a single topic using the MCP agent with retry logic
    """
    logger.info(f"Processing Reddit topic: {topic}")
    
    async with mcp_limiter:
        try:
            # Use MCP to search Reddit for relevant content
            tools = await load_mcp_tools(agent, {"reddit_search": True})
            agent_executor = create_react_agent(
                tools=tools,
                llm=ChatGoogleGenerativeAI(
                    model="gemini-1.0-pro",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3
                )
            )
            
            query = f"Search Reddit for discussions about {topic} from the last two weeks"
            result = await agent_executor.ainvoke({"input": query})
            
            if "overloaded" in str(result).lower():
                raise MCPOverloadedError("MCP service is overloaded")
            
            # Summarize Reddit content
            summary = gemini_summarize_reddit(topic, str(result))
            return summary
            
        except Exception as e:
            logger.error(f"Error processing topic '{topic}': {str(e)}")
            return f"Error processing topic '{topic}': {str(e)}"

async def scrape_reddit_topics(topics: List[str]) -> Dict[str, Any]:
    """
    Scrape Reddit content for multiple topics
    """
    if not topics:
        logger.warning("No topics provided for Reddit scraping")
        return {"reddit_analysis": {}, "errors": ["No topics provided"]}
    
    logger.info(f"Starting Reddit scraping for {len(topics)} topics: {topics}")
    
    try:
        async with stdio_client(server_params) as (read, write):
            agent = (read, write)
            
            reddit_analysis = {}
            errors = []
            successful_topics = 0
            
            # Process each topic
            for topic in topics:
                try:
                    result = await process_topic(agent, topic)
                    if not result.startswith("Error:"):
                        reddit_analysis[topic] = result
                        successful_topics += 1
                    else:
                        errors.append(result)
                except Exception as e:
                    error_msg = f"Failed to process topic '{topic}': {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            return {
                "reddit_analysis": reddit_analysis,
                "total_topics": len(topics),
                "successful_topics": successful_topics,
                "failed_topics": len(topics) - successful_topics,
                "errors": errors,
                "timestamp": datetime.now().isoformat(),
                "date_filter": two_weeks_ago_str
            }
    except Exception as e:
        error_msg = f"Failed to initialize Reddit scraping: {str(e)}"
        logger.error(error_msg)
        return {
            "reddit_analysis": {},
            "total_topics": len(topics),
            "successful_topics": 0,
            "failed_topics": len(topics),
            "errors": [error_msg],
            "timestamp": datetime.now().isoformat(),
            "date_filter": two_weeks_ago_str
        }

# Utility function for testing
async def test_reddit_scraping():
    """Test function for debugging"""
    test_topics = ["artificial intelligence", "climate change"]
    result = await scrape_reddit_topics(test_topics)
    print("Test Results:")
    print(f"Successful topics: {result['successful_topics']}")
    print(f"Failed topics: {result['failed_topics']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    
    for topic, analysis in result['reddit_analysis'].items():
        print(f"\n--- {topic} ---")
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_reddit_scraping())