import json
from datetime import datetime, timezone
from app.core.config import get_settings
from app.models import FeedbackRequest


def save_feedback(item: FeedbackRequest) -> None:
    path = get_settings().feedback_path
    row = item.model_dump()
    row["created_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
