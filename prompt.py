from datetime import datetime
from config import COUNTRY, LANGUAGE

def build_prompt(trend):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    news = "\nARTICLES ISSUS DE GOOGLE ACTUALITÉS :\n"
    for i, item in enumerate(trend.get("news", []), 1):
        news += f"""
ARTICLE {i}
Titre : {item.get('title','')}
Source : {item.get('source','')}
Publié : {item.get('published','')}

Résumé :
{item.get('summary','')}

Texte complet :
{item.get('content','')[:5000]}
---
"""

    return f"""
Tu es un journaliste professionnel expérimenté basé en {COUNTRY}.
Rédige EXCLUSIVEMENT en {LANGUAGE}.

DATE ET HEURE ACTUELLES :
{current_datetime}

SUJET PRINCIPAL :
{trend["title"]}

{news}

LIMITE FACTUELLE :
L'article doit être basé EXCLUSIVEMENT sur les articles Google Actualités
fournis.

- N'invente aucun fait.
- N'utilise aucune connaissance externe.
- Ne transforme jamais une supposition en fait.
- N'ajoute aucun nom, chiffre, date, heure, citation, résultat ou événement
  qui ne soit pas étayé par les sources.
- Reprends exactement les chiffres, pourcentages, changements, dates, heures,
  résultats et classements présents dans les sources.
- Ne fais aucun calcul toi-même.
- Un sondage n'est pas un résultat électoral ni une prédiction.
- Une intention politique n'est pas un résultat.
- N'affirme aucune relation de cause à effet si les sources ne l'établissent pas.
- Si une information n'est pas clairement étayée, supprime-la.
- Si les sources se contredisent, ne devine pas ; utilise uniquement ce qui est
  clairement confirmé.

SUJET CONCRET :
Traite uniquement la nouvelle concrète du SUJET PRINCIPAL.
N'ajoute pas d'autres nouvelles, de cas similaires ou d'informations
périphériques uniquement pour allonger l'article.
Chaque section doit appartenir au même événement ou développement.


============================================================
DATE DE PUBLICATION != DATE DE L'ÉVÉNEMENT - RÈGLE OBLIGATOIRE
============================================================

La date « Publié » ou la date de publication/mise à jour d'un article Google
Actualités n'est PAS automatiquement la date de l'événement rapporté.

- N'utilise jamais la date de publication comme date de l'événement simplement
  parce qu'elle est la date la plus récente affichée dans la source.
- Une date d'événement ne peut être utilisée que si le texte de l'article ou
  une indication explicite de la source confirme quand l'événement a réellement
  eu lieu.
- Si une source est publiée le 10 août mais rapporte un événement survenu le
  4 août, le 4 août est la date de l'événement et le 10 août est uniquement la
  date de publication.
- Distingue toujours la date de publication, la date de mise à jour, la date de
  l'événement, la date d'une décision et la date d'une annonce.
- Cette distinction est obligatoire notamment pour les décisions réglementaires
  ou judiciaires, naissances, décès, accidents, événements sportifs,
  lancements et situations similaires.
- Lorsque plusieurs dates apparaissent, attribue chaque date à son événement
  précis avant de l'utiliser dans l'article.
- Si la date de l'événement n'est pas clairement étayée par le texte de la
  source, n'affirme aucune date pour l'événement.
- Une date qui apparaît uniquement dans le champ « Publié » ne constitue JAMAIS
  une preuve suffisante que l'événement a eu lieu à cette date.

DATES / STATUT / SPORT :
Utilise la date et l'heure actuelles comme référence.
Ne présente jamais comme à venir un événement qui a déjà eu lieu.
S'il a déjà eu lieu, écris au passé et utilise le résultat final uniquement
s'il est confirmé par les sources.

Pour le sport :
- ne mentionne la compétition que si elle est confirmée ;
- n'invente aucun résultat, composition, blessure, entraîneur, forme ou pronostic ;
- ne transforme jamais un match amical ou de pré-saison en compétition officielle
  sans confirmation explicite.

NOMS PROPRES :
Les noms de personnes, clubs, équipes, organisations, entreprises, marques,
compétitions et lieux doivent être repris exactement comme dans les sources.
Ne les traduis pas, ne les corrige pas et ne les devine pas.

STYLE :
Français naturel, professionnel et journalistique.
Clair, objectif, précis, sans clickbait, spéculation, répétitions ou phrases
de remplissage.

Si les sources contiennent peu de matière fiable, écris un article plus court.
N'ajoute jamais de contenu pour atteindre une longueur ou un nombre de sections.

STRUCTURE :
Introduction.
De 1 à 5 sections selon la quantité de matière réellement étayée.
3 à 5 sections lorsque les sources fournissent suffisamment d'informations.
1 ou 2 sections sont parfaitement acceptables lorsque les sources ne permettent
pas d'ajouter davantage d'informations fiables.
Ne crée pas une section uniquement pour atteindre un nombre minimum.
Conclusion courte et factuelle, sans nouveaux faits.

Retourne EXCLUSIVEMENT un JSON valide :
{{
  "title": "",
  "description": "",
  "h1": "",
  "intro": "",
  "sections": [
    {{"title": "", "text": ""}}
  ]
}}

VÉRIFICATION FINALE :
- vérifie chaque chiffre, date, heure, nom et affirmation factuelle contre les sources ;
- pour chaque date importante, distingue explicitement la date de publication/
  mise à jour de la source et la date réelle de l'événement ;
- ne traite jamais une date « Publié » comme la date de l'événement sauf si le
  texte de la source confirme explicitement que l'événement a eu lieu à cette date ;
- n'ajoute aucun fait nouveau dans l'introduction, les transitions ou la conclusion ;
- supprime toute phrase douteuse ;
- retourne uniquement le JSON.
"""
