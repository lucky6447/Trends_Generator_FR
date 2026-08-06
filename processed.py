import json

from config import PROCESSED_FILE


def load_processed():

    if not PROCESSED_FILE.exists():
        return set()

    return set(
        json.loads(
            PROCESSED_FILE.read_text(
                encoding="utf-8"
            )
        )
    )


def save_processed(processed):

    PROCESSED_FILE.write_text(
        json.dumps(
            sorted(processed),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def is_processed(keyword, processed):

    return keyword in processed


def add_processed(keyword, processed):

    processed.add(keyword)
    save_processed(processed)