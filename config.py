from pathlib import Path

ROOT = Path(__file__).parent

# Google Trends France
RSS_URL = "https://trends.google.com/trending/rss?geo=FR"

# AI Article Generation
COUNTRY = "France"
LANGUAGE = "fr"

# Ollama Model
MODEL = "qwen2.5:14b"

# Website
SITE_NAME = "Tendances en France"
SITE_URL = "https://fr.trendcurrent.today"

# Directories
TREND_DIR = ROOT / "trends"
TEMPLATE_FILE = ROOT / "template.html"
PROCESSED_FILE = ROOT / "processed.json"

# Generator Settings
CHECK_INTERVAL = 600
MAX_ARTICLES_PER_RUN = 1

# Number of articles displayed on each index page
ARTICLES_PER_PAGE = 20