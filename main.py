"""
Runs the API for the card scanner.
"""

from api import app
import uvicorn

from config import settings
from logging_config import setup_logging

if __name__ == "__main__":
    # Initialize structured logging
    setup_logging(level=settings.log_level, json_format=settings.log_json)

    # Run the server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
