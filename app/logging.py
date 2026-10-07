import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "metrics", {}))
        return json.dumps(entry)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("zchat")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
