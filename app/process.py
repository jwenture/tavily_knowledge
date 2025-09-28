import sys
from pathlib import Path
from app.log import logger
# Add the parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from app.tavily_connection import Tavily_ArXiv
from app.arxiv_knowledge_graph import ArXivKnowledgeGraph

class ProcessFlow:
    def __init__(self, tavily_api_key: str, openai_api_key: str):
        self.tavily_client = Tavily_ArXiv(api_key=tavily_api_key)
        self.graph_generator = ArXivKnowledgeGraph(openai_api_key=openai_api_key)

    def search_and_process(self, query: str):
        # 1. Perform Tavily search (ArXiv only)
        search_results = self.tavily_client.search_arxiv(query=query)
        
        # 2. Extract paper data
        results = search_results.get("results", [])
        
        # 3. Generate knowledge graph
        self.graph_generator.build_knowledge_graph(papers=results)
        self.graph_generator.save_graph()
        self.graph_generator.create_html_graph()
        logger.info("Graph generated successfully")
        
        return {"status": "success", "message": "Graph generated successfully"}
    
    def test_process(self, query: str):
        response = self.tavily_client.search_arxiv_mock(query=query)
        results = response.get("results", [])
        if not results:
            return {"status": "error", "message": "No results found"}
        
        self.graph_generator.build_knowledge_graph(papers=results)

        ## Query the graph
        print("\nEntities related to 'Transformer':")
        related = self.graph_generator.query_graph("Transformer")
        for entity in related:
            print(f"  - {entity}")
        
        # Visualize the graph
        #kg_builder.visualize_graph("arxiv_knowledge_graph.png")
        
        # Save the graph
        self.graph_generator.save_graph(self.graph_generator.saved_graph_path)
        
        # Print graph statistics
        print(f"\nGraph Statistics:")
        print(f"Number of nodes: {self.graph_generator.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph_generator.graph.number_of_edges()}")
        self.graph_generator.create_html_graph()
        


if __name__ == "__main__":
    pflow = ProcessFlow(tavily_api_key="dfdf", openai_api_key="iliketesting")
    response = pflow.test_process(query="test")
    print(response)