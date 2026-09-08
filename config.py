from pathlib import Path

ROOT = Path(__file__).parent

# ===== Fixed language identity (do not inherit process environment) =====
LANGUAGE = 'fr'
LANGUAGE_NAME = 'Français'
COUNTRY = 'France'

# ===== Permanent discovery mode =====
SOURCE_FIRST = True
RSS_URL = "https://trends.google.com/trending/rss?geo=FR"

# ===== Ollama / site =====
MODEL = "ministral-3:14b-instruct-2512-q4_K_M"
SITE_NAME = 'TrendCurrent'
SITE_URL = 'https://trendcurrent.today'

# ===== Storage =====
TREND_DIR = ROOT / "trends"
TEMPLATE_FILE = ROOT / "template.html"
PROCESSED_FILE = ROOT / "processed.json"

# ===== Runner =====
CHECK_INTERVAL = 600
MAX_ARTICLES_PER_RUN = 1
ARTICLES_PER_PAGE = 18
