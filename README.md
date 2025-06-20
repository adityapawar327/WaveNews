# News Agent

A multi-source news and discussion aggregator with AI-powered summarization. Scrapes and summarizes news from Reddit and Twitter (X.com) using MCP and Gemini, with a Streamlit frontend and FastAPI backend.

## Features
- Scrape and summarize news from Reddit and Twitter (X.com)
- Uses MCP for robust web scraping
- Summarization powered by Google Gemini (via LangChain)
- Async, rate-limited scraping for reliability
- Modular codebase: separate scrapers for each source
- Streamlit frontend and FastAPI backend

## Setup
1. Clone the repository and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Set up your `.env` file with the following variables:
   - `GEMINI_API_KEY`
   - `API_TOKEN`
   - `WEB_UNLOCKER_ZONE`
   - `MCP_API_KEY` (for x_scrapper)

## Usage
- Run the backend:
  ```sh
  python backend.py
  ```
- Run the frontend:
  ```sh
  streamlit run frontend.py
  ```
- Test scrapers directly:
  ```sh
  python reddit__scraper.py
  python x_scrapper.py
  ```

## File Structure
- `reddit__scraper.py` — Reddit scraping and summarization
- `x_scrapper.py` — Twitter/X.com scraping and summarization
- `news_scraper.py` — (Other news sources)
- `models.py`, `utils.py` — Shared models and utilities
- `backend.py`, `frontend.py` — API and UI

## License
MIT
