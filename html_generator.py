from jinja2 import Environment,FileSystemLoader
from pathlib import Path
from config import ROOT,TREND_DIR,SITE_URL,LANGUAGE
env=Environment(loader=FileSystemLoader(ROOT/'templates'),autoescape=True)
template=env.get_template('article.html')
def render_article(article):
    return template.render(article=article,site_url=SITE_URL,language=LANGUAGE)
def save_article(slug,html):
    (TREND_DIR/f'{slug}.html').write_text(html,encoding='utf-8')
