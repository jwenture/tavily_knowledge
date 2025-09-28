
from tavily import TavilyClient
import re 
import json
from app.log import logger
_URL_PATTERN = re.compile(r'\([^)]*https?://[^)]*\)')
_CITATION_PATTERN = re.compile(r'\[\d+\]')
_REFERENCE_PATTERN = re.compile(
    r'^\*\s+.*?\[\d{4}[a-z]?\]\s+.*?\[.*?\]\s*\(https?://[^\)]+\)\s*\.$',
    re.MULTILINE
)
_REFERENCE_BULLET_PATTERN = re.compile(
    r'^\*\s+[A-Za-z].*?(?:\d{4}|\.\s*$)', 
    re.MULTILINE
)
_EMPTY_LIST_PATTERN = re.compile(r'\[\s*(,\s*)*\]')

class Tavily_ArXiv:
    def __init__(self, api_key: str = "tvly-dev-********************************"):
        self.client = TavilyClient(api_key)
        
    def search_arxiv(self, query: str):    
        logger.info(f"Search query: {query}")
       
        response = self.client.search(
            query = query,
            search_depth= "advanced",
            include_raw_content="markdown",
            chunks_per_source=1,
            max_results=3,
            include_domains=["https://arxiv.org/"]
        )
        with open('response_dump.json', 'w', encoding='utf-8') as f:
            json.dump(response, f, default=lambda x: x.__dict__, ensure_ascii=False, indent=2)

        with open("response_dump.json", 'r', encoding='utf-8') as f:
            response = json.load(f)

        for i, val in enumerate(response.get("results", [])):
            response["results"][i]["raw_content"] = self.clean_arxiv_text(val["raw_content"])
        logger.info(f"Query : {query} \n Total results: {len(response.get('results', []))}")
        return response
    
    
    def search_arxiv_mock(self, query: str):
        with open("app/mock_tavily.json", 'r', encoding='utf-8') as f:
            response = json.load(f)

        for i, val in enumerate(response.get("results", [])):
            response["results"][i]["raw_content"] = self.clean_arxiv_text(val["raw_content"])

        return response

    def clean_arxiv_text(self, text: str) -> str:
        text = _REFERENCE_PATTERN.sub('', text)
        text = _REFERENCE_BULLET_PATTERN.sub('', text)
        text  = _URL_PATTERN.sub('', text)
        text = _CITATION_PATTERN.sub('', text)
        text = _EMPTY_LIST_PATTERN.sub('', text)
        
        return text