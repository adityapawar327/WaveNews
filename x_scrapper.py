# x_scrapper.py
# Scrapes news from x.com using MCP, similar to other scrapers.

from langchain_mcp_adapters import MCPClient
from models import NewsArticle
from utils import clean_text
import os
from typing import List, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta
from mcp import StdioServerParameters
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

MCP_API_KEY = os.getenv("MCP_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

two_weeks_ago = datetime.today() - timedelta(days=14)
two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')

class MCPOverloadedError(Exception):
    pass

class TwitterScrapingError(Exception):
    pass

# Rate limiter: 1 request per 15 seconds
mcp_limiter = AsyncLimiter(1, 15)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")
WEB_UNLOCKER_ZONE = os.getenv("WEB_UNLOCKER_ZONE")

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

def gemini_summarize_twitter(topic: str, twitter_content: str) -> str:
    if not twitter_content or twitter_content.strip() == "":
        logger.warning(f"Empty Twitter content for topic: {topic}")
        return f"No Twitter content found for topic '{topic}'"
    
    system_prompt = f"""
You are a Twitter analysis expert. Analyze the provided Twitter (X.com) content about '{topic}' and create a comprehensive summary.

Requirements:
- Focus only on posts from after {two_weeks_ago_str}
- Provide clear, factual analysis
- Maintain objectivity and avoid bias
- Quote interesting tweets without mentioning usernames
"""

    user_prompt = f"""
Analyze the following Twitter (X.com) posts about '{topic}' and provide a comprehensive summary including:

1. Main discussion points and themes
2. Key opinions and perspectives expressed
3. Notable trends or patterns in the discussions
4. Interesting quotes from tweets (anonymized)
5. Overall sentiment analysis (positive/neutral/negative)
6. Community engagement level and discussion quality

Twitter Content:
{twitter_content[:8000]}  # Limit content to avoid token limits
"""

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            max_tokens=1500,
            timeout=30
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
    wait=wait_exponential(multiplier=2, min=15, max=120),
    retry=retry_if_exception_type((MCPOverloadedError, ConnectionError)),
    reraise=True
)
async def process_topic(agent, topic: str) -> str:
    logger.info(f"Processing Twitter topic: {topic}")
    async with mcp_limiter:
        try:
            tools = await load_mcp_tools(agent, {"x_search": True})
            agent_executor = create_react_agent(
                tools=tools,
                llm=ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GEMINI_API_KEY,
                    temperature=0.3
                )
            )
            query = f"Search Twitter (X.com) for discussions about {topic} from the last two weeks"
            result = await agent_executor.ainvoke({"input": query})
            if "overloaded" in str(result).lower():
                raise MCPOverloadedError("MCP service is overloaded")
            summary = gemini_summarize_twitter(topic, str(result))
            return summary
        except Exception as e:
            logger.error(f"Error processing topic '{topic}': {str(e)}")
            return f"Error processing topic '{topic}': {str(e)}"

async def scrape_twitter_topics(topics: List[str]) -> Dict[str, Any]:
    if not topics:
        logger.warning("No topics provided for Twitter scraping")
        return {"twitter_analysis": {}, "errors": ["No topics provided"]}
    logger.info(f"Starting Twitter scraping for {len(topics)} topics: {topics}")
    try:
        async with stdio_client(server_params) as (read, write):
            agent = (read, write)
            twitter_analysis = {}
            errors = []
            successful_topics = 0
            for topic in topics:
                try:
                    result = await process_topic(agent, topic)
                    if not result.startswith("Error:"):
                        twitter_analysis[topic] = result
                        successful_topics += 1
                    else:
                        errors.append(result)
                except Exception as e:
                    error_msg = f"Failed to process topic '{topic}': {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            return {
                "twitter_analysis": twitter_analysis,
                "total_topics": len(topics),
                "successful_topics": successful_topics,
                "failed_topics": len(topics) - successful_topics,
                "errors": errors,
                "timestamp": datetime.now().isoformat(),
                "date_filter": two_weeks_ago_str
            }
    except Exception as e:
        error_msg = f"Failed to initialize Twitter scraping: {str(e)}"
        logger.error(error_msg)
        return {
            "twitter_analysis": {},
            "total_topics": len(topics),
            "successful_topics": 0,
            "failed_topics": len(topics),
            "errors": [error_msg],
            "timestamp": datetime.now().isoformat(),
            "date_filter": two_weeks_ago_str
        }

# Utility function for testing
async def test_twitter_scraping():
    test_topics = ["artificial intelligence", "climate change"]
    result = await scrape_twitter_topics(test_topics)
    print("Test Results:")
    print(f"Successful topics: {result['successful_topics']}")
    print(f"Failed topics: {result['failed_topics']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    for topic, analysis in result['twitter_analysis'].items():
        print(f"\n--- {topic} ---")
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)

if __name__ == "__main__":
    asyncio.run(test_twitter_scraping())
