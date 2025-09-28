import logging
from contextlib import asynccontextmanager
from app.connect_db import MongoHandler
logger = logging.getLogger(__name__)  # Use a specific name
from dotenv import load_dotenv
import os

load_dotenv()
MONGO_SERVER = 'localhost:27017'
MONGO_CONNECTION_STRING = f"mongodb://usertesting:passtesting@{MONGO_SERVER}?directConnection=true"
MONGO_LOGGING_DATABASE = "logging_db"
MONGO_COLLECTION = "logs"

def setup_logging():
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        app_logger.removeHandler(handler)

    # Create and configure MongoDB handler
    mongo_handler = MongoHandler(
        connection_string=MONGO_CONNECTION_STRING,
        database=MONGO_LOGGING_DATABASE,
        collection=MONGO_COLLECTION
    )
    mongo_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    mongo_handler.setFormatter(formatter)

    # Add handler to logger
    app_logger.addHandler(mongo_handler)

    # Also add a console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)
    
    return app_logger, mongo_handler

logger, mongo_handler = setup_logging()

@asynccontextmanager
async def lifespan(app):
    # Startup code
    logger.info("FastAPI application starting up")
    yield
    # Shutdown code
    logger.info("FastAPI application shutting down")
    try:
        if hasattr(mongo_handler, 'client') and mongo_handler.client:
            mongo_handler.close()
    except Exception as e:
        logger.error(f"Error closing MongoDB handler: {e}")


