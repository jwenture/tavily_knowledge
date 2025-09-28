from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from typing import List, Dict, Any, Optional
import logging
from queue import Queue, Empty
import threading
import time
from datetime import datetime
from pymongo.errors import ServerSelectionTimeoutError

class MongoHandler(logging.Handler):
    def __init__(self, connection_string: str, database: str, collection: str):
        super().__init__()
        self.connection_string = connection_string
        self.database_name = database
        self.collection_name = collection
        self.client = None
        self.db = None
        self.collection = None
        self._connect()

    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            self.client.server_info()
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
    

    
    def emit(self, record):
        try:
            if self.client is None:
                self._connect()

            # Format the log record
            log_entry = {
                'timestamp': datetime.utcnow(),
                'level': record.levelname,
                #'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                #'pathname': record.pathname,
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = self.formatException(record.exc_info)
            
            # Add extra fields if any
            if hasattr(record, 'extra_data'):
                log_entry.update(record.extra_data)
            
            # Insert into MongoDB
            self.collection.insert_one(log_entry)
            
        except Exception as e:
            # Handle logging errors to avoid infinite recursion
            print(f"Error in MongoHandler: {e}")
    
    def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None

        super().close()


class AsyncMongoHandler(logging.Handler):
    def __init__(self, connection_string: str, database: str, collection: str, 
                 batch_size: int = 10, flush_interval: float = 1.0):
        super().__init__()
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Thread-safe queue for log records
        self.log_queue = Queue()
        self.stop_event = threading.Event()
        
        # Start background thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def emit(self, record):
        # Convert record to dict immediately to avoid issues with thread safety
        log_entry = self._format_record(record)
        self.log_queue.put(log_entry)
    
    def _format_record(self, record):
        log_entry = {
            'timestamp': datetime.utcnow(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'pathname': record.pathname,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 
                          'stack_info', 'message']:
                log_entry[f'extra_{key}'] = str(value)
        
        return log_entry
    
    def _worker(self):
        client = None
        try:
            client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            db = client[self.database]
            collection = db[self.collection]
            
            batch = []
            last_flush = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Get log record with timeout
                    record = self.log_queue.get(timeout=0.1)
                    batch.append(record)
                    
                    # Flush if batch is full or timeout reached
                    current_time = time.time()
                    if (len(batch) >= self.batch_size or 
                        current_time - last_flush >= self.flush_interval):
                        if batch:
                            try:
                                collection.insert_many(batch)
                                batch = []
                                last_flush = current_time
                            except Exception as e:
                                print(f"Error inserting logs to MongoDB: {e}")
                                # Optionally retry or handle differently
                
                except Empty:
                    # Check if we need to flush due to timeout
                    if batch and time.time() - last_flush >= self.flush_interval:
                        try:
                            collection.insert_many(batch)
                            batch = []
                            last_flush = time.time()
                        except Exception as e:
                            print(f"Error inserting logs to MongoDB: {e}")
                    continue
                
                except Exception as e:
                    print(f"Error in log worker: {e}")
        
        except ServerSelectionTimeoutError:
            print("Could not connect to MongoDB for logging")
        except Exception as e:
            print(f"Unexpected error in log worker: {e}")
        finally:
            if client:
                client.close()
    
    def close(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        super().close()

class KnowledgeGraphDBConnector:
    def __init__(self, uri: str = "mongodb://localhost:27017/", db_name: str = "knowledge_graph"):
        logging.info(f"Attempting to connnect to MongoDB at {uri}")
        print(f"Attempting to connnect to MongoDB at {uri}")
        self.client = MongoClient(uri)
        
        self.client.admin.command('ping')
        logging.info(f"Connected to MongoDB at {uri}")
        print(f"Connected to MongoDB at {uri}")
        self.db = self.client[db_name]
        self.nodes = self.db.nodes
        self.edges = self.db.edges
        self._ensure_indexes()

    def _ensure_indexes(self):
        # Ensure uniqueness on node id
        self.nodes.create_index("id", unique=True)
        # Optional: text index for title (for keyword search)
        self.nodes.create_index([("title", "text")])
        # If using vector search, you must manually create a vector index in Atlas UI or via CLI
        # This script assumes 'embedding' field exists

    def insert_graph(self, graph: Dict[str, List]):
        """Insert nodes and edges from JSON graph."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        print(nodes)
        # Insert nodes
        for node in nodes:
            try:
                # Optional: compute embedding here
                # text_for_embedding = " ".join(node.get("title", [])) + " " + node.get("type", "")
                # node["embedding"] = embedding_model.encode(text_for_embedding).tolist()
                self.nodes.insert_one(node)
            except DuplicateKeyError:
                logging.warning(f"Node {node['id']} already exists. Skipping.")

        # Insert edges (with validation that source/target exist)
        valid_edges = []
        node_ids = set(doc["id"] for doc in self.nodes.find({}, {"id": 1}))
        for edge in edges:
            if edge["source"] not in node_ids:
                logging.warning(f"Skipping edge: source node {edge['source']} not found.")
                continue
            if edge["target"] not in node_ids:
                logging.warning(f"Skipping edge: target node {edge['target']} not found.")
                continue
            valid_edges.append(edge)

        if valid_edges:
            self.edges.insert_many(valid_edges)

    def delete_node(self, node_id: str):
        """Delete a node and all edges connected to it (in/out)."""
        result_nodes = self.nodes.delete_one({"id": node_id})
        result_edges = self.edges.delete_many({
            "$or": [
                {"source": node_id},
                {"target": node_id}
            ]
        })
        logging.info(f"Deleted {result_nodes.deleted_count} node(s) and {result_edges.deleted_count} edge(s).")

    def hybrid_search(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        text_weight: float = 0.5,
        vector_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search: combine text relevance and vector similarity.
        Requires precomputed 'embedding' field in nodes.
        """
        if not query_text and not query_vector:
            raise ValueError("At least one of query_text or query_vector must be provided.")

        pipeline = []

        # Stage 1: Vector search (if vector provided)
        if query_vector:
            pipeline.append({
                "$vectorSearch": {
                    "index": "vector_index",  # ← Must match your Atlas vector index name
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": top_k * 2  # get more for reranking
                }
            })
            pipeline.append({
                "$addFields": {
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            })

        # Stage 2: Text search (if text provided)
        if query_text:
            text_stage = {
                    "$match": {
                        "$or": [
                            { "title": { "$regex": query_text, "$options": "i" } },
                            { "name": { "$regex": query_text, "$options": "i" } }
                        ]
                }
            }
            if query_vector:
                # Merge vector and text results via unionWith
                text_pipeline = [
                    text_stage,
                    {"$limit": top_k * 2},
                    {"$addFields": {"text_score": {"$meta": "searchScore"}}}
                ]
                pipeline = [
                    {"$limit": 0},
                    {"$unionWith": {"coll": "nodes", "pipeline": pipeline}},
                    {"$unionWith": {"coll": "nodes", "pipeline": text_pipeline}},
                    {"$group": {
                        "_id": "$id",
                        "doc": {"$first": "$$ROOT"},
                        "max_vec_score": {"$max": "$vector_score"},
                        "max_text_score": {"$max": "$text_score"}
                    }},
                    {"$replaceRoot": {
                        "newRoot": {
                            "$mergeObjects": [
                                "$doc",
                                {"vector_score": "$max_vec_score"},
                                {"text_score": "$max_text_score"}
                            ]
                        }
                    }}
                ]
            else:
                pipeline = [
                    text_stage,
                    {"$limit": top_k},
                    {"$addFields": {"text_score": {"$meta": "searchScore"}}}
                ]

        # Stage 3: Compute hybrid score and sort
        if query_text and query_vector:
            pipeline.append({
                "$addFields": {
                    "hybrid_score": {
                        "$add": [
                            {"$multiply": ["$vector_score", vector_weight]},
                            {"$multiply": ["$text_score", text_weight]}
                        ]
                    }
                }
            })
            pipeline.append({"$sort": {"hybrid_score": -1}})
        elif query_vector:
            pipeline.append({"$sort": {"vector_score": -1}})
        elif query_text:
            pipeline.append({"$sort": {"text_score": -1}})

        pipeline.append({"$limit": top_k})
        return list(self.nodes.aggregate(pipeline))

    def close(self):
        self.client.close()


# Example usage
if __name__ == "__main__":
    import json
    MONGO_SERVER = 'localhost:27017'
    MONGO_DATABASE = "your_database_name"
    MONGO_COLLECTION = "logs"

    # Load your graph
    with open("knowledge_graph.json", "r") as f:
        graph = json.load(f)

    kg_db = KnowledgeGraphDBConnector(uri=f"mongodb://usertesting:passtesting@{MONGO_SERVER}?directConnection=true", 
                                      db_name="knowledge_graph_test")  # or your Atlas URI

    # Insert graph
    #kg_db.insert_graph(graph)

    # Example: delete a node safely
    # kg_db.delete_node("Superconductivity")

    # Example: hybrid search (you must provide embedding if using vector)
    results = kg_db.hybrid_search(query_text="reinforce")#, query_vector=[0.1, -0.5, ...])
    print(results)

    kg_db.close()