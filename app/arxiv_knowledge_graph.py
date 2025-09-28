import os
import re
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import networkx as nx
import matplotlib.pyplot as plt
import json
from pyvis.network import Network
from app.customjs import custom_js
from app.log import logger
from dotenv import load_dotenv

load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BASE_URL = os.environ['BASE_URL']
MODEL_NAME= os.environ['MODEL_NAME']

CHUNK_STOP = 2
DOCUMENTS_STOP =6

class TextAnalysisResponse(BaseModel):
    summary: str = Field(
        default="",
        description="Brief summary of the content"
    )

    entities: List[str] = Field(
        default_factory=list,
        description="List of key entities/concepts mentioned in the text"
    )
    relations: List[Tuple[str, str, str]] = Field(
        default_factory=list,
        description="List of relations between entities in format (entity1, relation, entity2)"
    )

class ArXivKnowledgeGraph:
    def __init__(self, openai_api_key: str, saved_graph_path: str = "app/knowledge_graph.json"):
        """Initialize the knowledge graph builder"""
        self.openai_api_key = openai_api_key
        
        if MODEL_NAME == "qwen":
             self.llm = ChatOpenAI(
                temperature=0.1,
                model_name="qwen",
                openai_api_base=BASE_URL,
                openai_api_key=openai_api_key
            )
        else:
            self.llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-5-nano-2025-08-07",#"qwen",
                #openai_api_base=BASE_URL,
                openai_api_key=openai_api_key
            )
        self._test_connection()
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
            length_function=len,
        )
        self.saved_graph_path = saved_graph_path
        
        # Create knowledge graph
        self.load_graph(file_path=self.saved_graph_path)
        
        self.existing_nodes = list(self.graph.nodes)
        
    def _test_connection(self):
        """Test connection during initialization"""
        try:
            # Very simple test call
            test_response = self.llm.invoke("Say 'Hello' in one word. No thinking required.")
            #print("✅ LLM connected successfully!")
            #print(f"Test response: {test_response}")
        except Exception as e:
            #print(f"❌ Failed to connect to LLM: {e}")
            logger.error(f"Could not connect to LLM at {BASE_URL}")
            raise ConnectionError(f"Could not connect to LLM at {BASE_URL}") from e

    def preprocess_papers_mock(self, papers: List[str]) -> List[Document]:
        """Preprocess the arXiv papers into documents"""
        documents = []
        for i, paper_text in enumerate(papers):
            # Clean the text
            cleaned_text = self._clean_text(paper_text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            for j, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"paper_id": f"paper_{i}", "chunk_id": j}
                ))
        
        return documents
    

    def preprocess_papers(self, papers: List[str]) -> List[Document]:
        """Preprocess the arXiv papers into documents"""
        documents = []
        for i, paper_item in enumerate(papers):
            paper_raw_content = paper_item["raw_content"]
            paper_title = paper_item["title"]
            url = paper_item["url"]
            
            # Clean the text
            cleaned_text = self._clean_text(paper_raw_content)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            for j, chunk in enumerate(chunks):
                if j >= CHUNK_STOP:
                    break
                documents.append(Document(
                    page_content=chunk,
                    metadata={"title": paper_title, "url": url, "chunk_id": j}
                ))
        
        return documents

    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_entities_relations(self, document: Document) -> Dict[str, Any]:
        """Extract entities and relations from a document chunk using LLM"""
        model_with_structure = self.llm.with_structured_output(TextAnalysisResponse)
        existing_nodes_prompt=""
        if self.existing_nodes:
            existing_nodes_prompt = "Here are all the existing nodes, do not create new nodes if it is similar to the existing nodes, just reuse the existing nodes in the exact same format: " + ", ".join(self.existing_nodes)
            
        # Create prompt template
        prompt_template = f"""
            Analyze the following text from a research paper and extract:
            1. A brief summary
            2. Key entities/concepts (people, methods, theories, technologies, etc.)
            3. Relations between these entities in the format (entity1, relation, entity2)
            
            The entities and relations should be reasonably broad and general to describe the paper,
            systems, and general ideas. The total number of entities and relations should not exceed 5.
            {existing_nodes_prompt}
                 
            Text from the research paper: {document.page_content}
            """
        
        try:
            structured_output = model_with_structure.invoke(prompt_template)
            print(f"✅ Parsed Output: {structured_output}")  # DEBUG
            return structured_output.dict()
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            return {"entities": [], "relations": [], "summary": f"Error: {str(e)}"}
    
    def build_knowledge_graph(self, papers: List[str]):
        """Build knowledge graph from arXiv papers"""
        
        #documents = self.preprocess_papers_mock(papers)
        documents = self.preprocess_papers(papers)
        
        print(f"Extracting knowledge from {len(documents)} chunks...")
        logger.info(f"Extracting knowledge from {len(documents)} chunks...")
        for i, doc in enumerate(documents):
            if i >= DOCUMENTS_STOP:
                break
            
            logger.info(f"Processing chunk {i+1}/{len(documents)} | {doc.metadata}")
            # Extract entities and relations
            extracted_data = self.extract_entities_relations(doc)
            logger.info(f"Processing chunk {i+1}/{len(documents)} | {extracted_data}")
            # Add entities as nodes
            for entity_ in extracted_data.get("entities", []):
                entity = entity_.strip().lower().replace(" ", "_")
                if entity and isinstance(entity, str):
                    if entity not in self.graph:
                        self.graph.add_node(entity, type="concept", name=entity_, title=set(), url=set())
                    self.graph.nodes[entity]["title"].add(doc.metadata["title"])
                    self.graph.nodes[entity]["url"].add(doc.metadata["url"])
            
            # Add relations as edges
            for relation_tuple in extracted_data.get("relations", []):
                if (isinstance(relation_tuple, list) or isinstance(relation_tuple, tuple)) and len(relation_tuple) == 3:
                    entity1_, relation, entity2_ = relation_tuple
                    entity1 = entity1_.strip().lower().replace(" ", "_")
                    entity2 = entity2_.strip().lower().replace(" ", "_")
                    if entity1 and entity2 and relation:
                        if entity1 not in self.graph:
                            self.graph.add_node(entity1, type="concept", name=entity1_, title=set(), url=set())
                            self.graph.nodes[entity1]["title"].add(doc.metadata["title"])
                            self.graph.nodes[entity1]["url"].add(doc.metadata["url"])
                        if entity2 not in self.graph:
                            self.graph.add_node(entity2, type="concept", name=entity2_, title=set(), url=set())
                            self.graph.nodes[entity2]["title"].add(doc.metadata["title"])
                            self.graph.nodes[entity2]["url"].add(doc.metadata["url"])

                        # Add edge with relation type
                        if self.graph.has_edge(entity1, entity2):
                            # Update existing edge
                            self.graph[entity1][entity2]["relations"].add(relation)
                        else:
                            self.graph.add_edge(entity1, entity2, relations={relation})
                            
            self.existing_nodes = list(self.graph.nodes)
    
    def query_graph(self, entity: str, depth: int = 2) -> List:
        """Query the knowledge graph for entities related to a specific concept"""
        if entity not in self.graph:
            return []
        
        # Get neighbors up to specified depth
        related_entities = []
        for depth_level in range(1, depth + 1):
            for node in nx.single_source_shortest_path_length(self.graph, entity, depth_level).keys():
                if node != entity and node not in related_entities:
                    related_entities.append(node)
        
        return related_entities
    
    def visualize_graph(self, output_path: str = "knowledge_graph.png"):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(15, 10))
        
        # Use spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        # Draw edge labels (relations)
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            relations = data.get('relations', set())
            if relations:
                edge_labels[(u, v)] = ', '.join(list(relations)[:2])  # Show first 2 relations
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("arXiv Papers Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_graph(self, file_path: str = ""):
        """Save the graph to a file"""
        if not file_path:
            file_path = self.saved_graph_path
        """Save the graph to a file"""
        # Convert to JSON serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        for node, data in self.graph.nodes(data=True):
            graph_data["nodes"].append({
                "id": node,
                "name": data.get("name", node),
                "type": data.get("type", "concept"),
                "title": list(data.get("title", [])),
                "url": list(data.get("url", []))
            })
        
        for u, v, data in self.graph.edges(data=True):
            graph_data["edges"].append({
                "source": u,
                "target": v,
                "relations": list(data.get("relations", set()))
            })
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, file_path: str = ""):
        """Load the graph from a file"""
        if not file_path:
            file_path = self.saved_graph_path
        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)
        except:
            self.graph = nx.Graph()
            return
            
        self.graph = nx.Graph()
        
        for node_data in graph_data["nodes"]:
            self.graph.add_node(
                node_data["id"],
                type=node_data["type"],
                title=set(node_data["title"]),
                url=set(node_data["url"]),
            )
        
        for edge_data in graph_data["edges"]:
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                relations=set(edge_data["relations"])
            )
        print(self.graph.nodes)

    def process_graph_for_html(self):
        graph_out = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        for node, attrs in self.graph.nodes(data=True):
            serializable_attrs = {}
            for key, value in attrs.items():
                if key == "title" and "url" in attrs:
                    serializable_attrs[key] = serialize_title_url(attrs["title"], attrs["url"])
                elif key == "url" and "title" in attrs:
                    continue
                else:
                    serializable_attrs[key] = make_serializable(value)
            graph_out.add_node(node, **serializable_attrs)

        # Clean edge attributes  
        for u, v, attrs in self.graph.edges(data=True):
            serializable_attrs = {}
            for key, value in attrs.items():
                serializable_attrs[key] = make_serializable(value)
            graph_out.add_edge(u, v, **serializable_attrs)
        
        return graph_out

    def create_html_graph(self):
        serializable_graph = self.process_graph_for_html()
        directed = serializable_graph.is_directed()
        net = Network(notebook=False, directed=directed)
        net.from_nx(serializable_graph)
        net.set_options("""
            var options = {
                "physics": {
                    "enabled": true,
                    "stabilization": {"iterations": 10}
                },
                "nodes": {
                    "shape": "dot",
                    "size": 25,
                    "font": {
                    "size": 16 }
                },
                "edges": {
                    "smooth": {
                    "type": "continuous"
                    }
                }
            }
            """)

        net.write_html("static/output1.html")
        with open("static/output1.html", "r") as f:
            html_content = f.read()
            
        html_content = html_content.replace("</body>", custom_js + "</body>")
        with open("static/output1.html", "w") as f:
            f.write(html_content)
        import shutil

        shutil.move("static/output1.html", "static/output.html")



def serialize_title_url(set_of_titles, set_of_urls):
    list_of_titles = list(set_of_titles)
    list_of_urls = list(set_of_urls)
    result = []
    for i, title in enumerate(list_of_titles):
        if i < len(list_of_urls) and list_of_urls[i]:
            result.append(f'<a href="{list_of_urls[i]}" target="_blank">{title}</a><br/>')
        else:
            result.append(f'{title}<br/>')
    return ''.join(result)

def make_serializable(obj):
    """Convert common non-serializable types to JSON-friendly equivalents"""
    if isinstance(obj, (set, tuple)):
        return list(obj) #f"<b><a href='' target='_blank' >{list(obj)}</a></b>"
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj

# Example usage
def main():
    # Replace with your OpenAI API key
    OPENAI_API_KEY = "iliketesting"
    
    # Sample arXiv papers (replace with your actual papers)
    sample_papers = [
        """
        Title: Attention Is All You Need
        Authors: Vaswani et al.
        Abstract: We propose a new simple network architecture, the Transformer, 
        based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. 
        Experiments on two machine translation tasks show these models to be superior in quality 
        while being more parallelizable and requiring significantly less time to train.
        """,
        """
        Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        Authors: Devlin et al.
        Abstract: We introduce a new language representation model called BERT, which stands for 
        Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, 
        BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly 
        conditioning on both left and right context in all layers.
        """,
        """
        Title: Transformer Models for Natural Language Processing
        Abstract: Transformer architecture uses attention mechanisms. 
        BERT is based on Transformer architecture. 
        Attention allows models to focus on important words. 
        Vaswani invented the Transformer model.
        """,
        """
        Title: BERT and Language Understanding  
        Abstract: BERT uses bidirectional transformers. 
        Transformer architecture employs self-attention. 
        Devlin created BERT for language tasks.
        BERT improves upon original Transformer design.
        """
    ]
    
    # Initialize knowledge graph builder
    kg_builder = ArXivKnowledgeGraph(OPENAI_API_KEY)
    
    # Build knowledge graph
    kg_builder.build_knowledge_graph(sample_papers)

    # Query the graph
    print("\nEntities related to 'Transformer':")
    related = kg_builder.query_graph("Transformer")
    for entity in related:
        print(f"  - {entity}")
    
    # Visualize the graph
    kg_builder.visualize_graph("arxiv_knowledge_graph.png")
    
    # Save the graph
    kg_builder.save_graph(kg_builder.saved_graph_path)
    
    # Print graph statistics
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {kg_builder.graph.number_of_nodes()}")
    print(f"Number of edges: {kg_builder.graph.number_of_edges()}")
    print(f"Graph density: {nx.density(kg_builder.graph):.4f}")

def build_main():
    # Replace with your OpenAI API key
    OPENAI_API_KEY = "iliketesting"
    
    # Sample arXiv papers (replace with your actual papers)
    sample_papers = [
        """
        Link: https://google.com
        Title: Attention Is All You Need
        Authors: Vaswani et al.
        Abstract: We propose a new simple network architecture, the Transformer, 
        based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. 
        Experiments on two machine translation tasks show these models to be superior in quality 
        while being more parallelizable and requiring significantly less time to train.
        """,
        #"""
        #Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        #Authors: Devlin et al.
        #Abstract: We introduce a new language representation model called BERT, which stands for 
        #Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, 
        #BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly 
        #conditioning on both left and right context in all layers.
        #""",
        #"""
        #Title: Transformer Models for Natural Language Processing
        #Abstract: Transformer architecture uses attention mechanisms. 
        #BERT is based on Transformer architecture. 
        #Attention allows models to focus on important words. 
        #Vaswani invented the Transformer model.
        #""",
        #"""
        #Title: BERT and Language Understanding  
        #Abstract: BERT uses bidirectional transformers. 
        #Transformer architecture employs self-attention. 
        #Devlin created BERT for language tasks.
        #BERT improves upon original Transformer design.
        #"""
    ]
    
    # Initialize knowledge graph builder
    kg_builder = ArXivKnowledgeGraph(OPENAI_API_KEY)
    
    # Build knowledge graph
    kg_builder.build_knowledge_graph(sample_papers)

    ## Query the graph
    print("\nEntities related to 'Transformer':")
    related = kg_builder.query_graph("Transformer")
    for entity in related:
        print(f"  - {entity}")
    
    # Visualize the graph
    #kg_builder.visualize_graph("arxiv_knowledge_graph.png")
    
    # Save the graph
    kg_builder.save_graph(kg_builder.saved_graph_path)
    
    # Print graph statistics
    print(f"\nGraph Statistics:")
    print(f"Number of nodes: {kg_builder.graph.number_of_nodes()}")
    print(f"Number of edges: {kg_builder.graph.number_of_edges()}")
    print(f"Graph density: {nx.density(kg_builder.graph):.4f}")
    kg_builder.create_html_graph()
    
if __name__ == "__main__":
    build_main()