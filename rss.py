import feedparser

from config import RSS_URL


SPORT_KEYWORDS = {
    # Общи
    "sport", "sports", "match", "game", "fixture", "result", "score",
    "live", "playoff", "play-offs", "final", "semi-final", "quarter-final",
    "championship", "tournament", "season", "derby", "cup", "league",

    # Футбол
    "football", "soccer", "fc", "cf", "sc", "afc", "uefa", "fifa",
    "champions league", "europa league", "conference league",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "eredivisie", "liga mx", "mls",

    # Баскетбол
    "basketball", "nba", "wnba", "euroleague", "fiba",

    # Тенис
    "tennis", "atp", "wta", "grand slam", "wimbledon",
    "roland garros", "us open", "australian open",

    # Формула 1 / Моторни
    "formula 1", "formula one", "f1", "motogp", "indycar", "nascar",
    "rally", "motorsport",

    # Бойни спортове
    "ufc", "mma", "boxing", "boxing", "wrestling", "kickboxing",

    # Други
    "golf", "cycling", "tour de france", "cricket", "rugby",
    "baseball", "mlb", "hockey", "nhl", "volleyball",
    "handball", "badminton", "snooker", "darts",

    # Често срещани клубове
    "arsenal", "chelsea", "liverpool", "manchester", "united", "city",
    "tottenham", "barcelona", "real madrid", "atletico",
    "juventus", "inter", "milan", "napoli", "roma",
    "bayern", "dortmund", "psg", "ajax", "porto",
    "benfica", "celtic", "rangers", "galatasaray",
    "fenerbahce", "besiktas", "paok", "anderlecht"
}


def clean(value):
    if not value:
        return ""
    return " ".join(str(value).split())


def is_sport(title):
    title = title.lower()

    # "A vs B"
    if " vs " in title or " v " in title:
        return True

    return any(keyword in title for keyword in SPORT_KEYWORDS)


def fetch_trends():
    feed = feedparser.parse(RSS_URL)

    trends = []

    for item in feed.entries:
        title = clean(item.get("title"))

        if not title:
            continue

        if is_sport(title):
            continue

        trends.append({
            "title": title,
            "link": clean(item.get("link")),
            "published": clean(item.get("published")),
            "traffic": clean(item.get("ht_approx_traffic")),
        })

    return trends