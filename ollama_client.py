import json
import os
import re
from ollama import chat
from config import MODEL

def clean_json(text):
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    return match.group(0) if match else text.strip()

def _call(prompt, temperature=0.15, num_predict=4096):
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40,
            "num_ctx": 8192,
            "num_predict": num_predict,
            "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "16")),
            "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "0")),
        },
        format="json",
    )
    return json.loads(clean_json(response.message.content))

def _extract_facts(source_prompt):
    """
    First-pass source fact extraction.

    The ledger explicitly separates publication/update dates from event dates so
    downstream generation and validation do not have to infer that relationship
    repeatedly from the raw source prompt.
    """
    return _call(f"""
Tu es un extracteur de faits pour un système éditorial d'actualité en français.

Travaille UNIQUEMENT à partir des SOURCES fournies ci-dessous.
N'utilise aucune connaissance externe et ne déduis pas une date d'événement à
partir de la date de publication ou de mise à jour.

Construis un EVIDENCE LEDGER structuré. Pour chaque information temporelle,
distingue explicitement :
- publication_dates: dates de publication des sources
- update_dates: dates de mise à jour des sources
- event_dates: dates explicitement attribuées à un événement dans le texte
- relative_dates: expressions comme "hier", "jeudi dernier", "ces derniers jours"
  uniquement si elles apparaissent réellement dans la source
- event_status: à venir, en cours, terminé, ou inconnu, uniquement si le texte
  de la source permet de l'établir
- facts: faits importants avec leur formulation et leur support source

RÈGLE ABSOLUE :
Une date placée uniquement dans les métadonnées "Publié", "Mis à jour" ou
équivalent DOIT rester une publication_date/update_date et NE DOIT PAS être
copiée dans event_dates.

Si une date d'événement n'est pas explicitement confirmée, laisse event_dates
vide. Ne devine jamais.

Retourne UNIQUEMENT ce JSON :
{{
  "publication_dates": [],
  "update_dates": [],
  "event_dates": [],
  "relative_dates": [],
  "event_status": [],
  "facts": []
}}

SOURCES :
{source_prompt}
""", temperature=0.0, num_predict=3000)


def _generate_article(source_prompt, ledger):
    return _call(f"""
{source_prompt}

Tu dois utiliser uniquement les faits directement étayés par les sources.

EVIDENCE LEDGER :
{json.dumps(ledger, ensure_ascii=False, indent=2)}

RÈGLES :
- N'ajoute aucune connaissance externe, cause, motivation, prédiction ou
  information de contexte non étayée.
- Ne transforme JAMAIS une publication_date ou update_date en event_date.
- N'invente aucune date d'événement.
- Utilise une date d'événement uniquement si elle figure dans event_dates.
- Utilise une expression relative uniquement si elle figure dans relative_dates.
- Si aucune date d'événement n'est établie, il est parfaitement acceptable de
  ne pas dater l'événement.
- Respecte event_status du ledger. Ne présente pas un événement futur comme
  déjà arrivé ni un événement terminé comme à venir.
- Si plusieurs sources donnent des chiffres différents et qu'aucune version
  unique n'est confirmée, conserve l'incertitude au lieu de choisir un chiffre.
- Retourne uniquement le JSON demandé.
""", temperature=0.15, num_predict=4096)


def _validate(source_prompt, article, ledger):
    return _call(f"""
Tu es un vérificateur factuel strict d'un article d'actualité en français.

Compare l'ARTICLE uniquement aux SOURCES et à l'EVIDENCE LEDGER extrait des
sources. N'utilise aucune connaissance externe.

EVIDENCE LEDGER :
{json.dumps(ledger, ensure_ascii=False, indent=2)}

ARTICLE :
{json.dumps(article, ensure_ascii=False, indent=2)}

RÈGLE CRITIQUE — PUBLICATION VS ÉVÉNEMENT :

1. Une publication_date ou update_date du ledger n'est PAS une event_date.
2. Ne considère une date comme date d'événement que si :
   - elle apparaît explicitement dans l'article comme date de l'événement, ET
   - elle est confirmée par event_dates du ledger.
3. Si l'article NE CONTIENT PAS de date d'événement explicite, ne lui reproche
   PAS d'avoir utilisé la publication date simplement parce que la source a été
   publiée ce jour-là.
4. Une phrase factuelle sans date, par exemple « Un incendie a entraîné la
   fermeture de l'A7 », ne doit PAS être traitée comme si elle contenait la
   publication date.
5. Si event_dates est vide, l'article peut parfaitement être accepté sans date
   d'événement, à condition que ses autres affirmations soient étayées.
6. Une expression relative (« hier », « jeudi dernier », « ces derniers jours »)
   ne doit être acceptée que si elle figure dans relative_dates ou est clairement
   résolue par le texte source.
7. Une date d'événement incorrecte, une confusion publication/événement, ou un
   statut temporel incorrect est un FAIL.
8. Si plusieurs sources donnent des chiffres différents et qu'aucune source ne
   confirme une version unique, l'article doit conserver l'incertitude et ne
   doit pas choisir arbitrairement une valeur.

IMPORTANT :
Ne déduis PAS une date à partir du contexte, de la date de publication, de la
date système ou de la position d'une information dans la source.

Vérifie aussi les chiffres, pourcentages, noms, identités, sondages, résultats,
causalité, prédictions, intentions et le statut des événements.

Retourne UNIQUEMENT :
{{"passed": true, "errors": [], "corrections": []}}

Si une affirmation n'est pas étayée ou est matériellement déformée :
{{"passed": false, "errors": [...], "corrections": [...]}}

SOURCES :
{source_prompt}
""", temperature=0.0, num_predict=2500)


def _repair(source_prompt, article, validation, ledger):
    return _call(f"""
Répare cet article en français en utilisant UNIQUEMENT les sources et
l'EVIDENCE LEDGER fournis.

EVIDENCE LEDGER :
{json.dumps(ledger, ensure_ascii=False, indent=2)}

FACT CHECK :
{json.dumps(validation, ensure_ascii=False, indent=2)}

ARTICLE :
{json.dumps(article, ensure_ascii=False, indent=2)}

RÈGLES CRITIQUES :
- Une publication_date ou update_date n'est jamais une preuve de la date de
  l'événement.
- Si une date d'événement n'existe pas dans event_dates, supprime la date
  d'événement de l'article au lieu de deviner.
- Si l'article ne contient aucune date d'événement, ne lui ajoute PAS de date
  uniquement parce que la source a été publiée à cette date.
- N'ajoute jamais « aujourd'hui », « hier », « jeudi dernier », « ces derniers
  jours » ou une autre expression temporelle qui n'est pas étayée par le ledger.
- Ne modifie pas les chiffres, dates, noms ou résultats correctement confirmés.
- Si les sources présentent des chiffres contradictoires sans version unique
  confirmée, conserve une formulation explicitement incertaine.
- Corrige uniquement les erreurs signalées ou les affirmations non étayées.
- Si une section n'est plus étayée, supprime-la.
- Un article plus court est acceptable.
- N'ajoute aucune connaissance externe.
- Ne transforme jamais une publication_date en event_date.
- Respecte le statut temporel de l'événement.

Retourne uniquement le JSON de l'article.
""", temperature=0.05, num_predict=4096)


def generate(prompt, retries=1):
    # 1) Extract source-grounded facts once.
    ledger = _extract_facts(prompt)

    # 2) Generate from the evidence ledger.
    article = _generate_article(prompt, ledger)

    # 3) Independent fact check against sources + ledger.
    check = _validate(prompt, article, ledger)

    if check.get("passed") is True:
        return article

    print("[FACT CHECK FAILED]")
    print(json.dumps(check, ensure_ascii=False, indent=2))

    # 4) Repair only from the same evidence ledger.
    repaired = _repair(prompt, article, check, ledger)

    # 5) Final independent fact check.
    repair_check = _validate(prompt, repaired, ledger)

    if repair_check.get("passed") is True:
        print("[FACT CHECK PASSED AFTER REPAIR]")
        return repaired

    print("[FACT CHECK FAILED AFTER REPAIR]")
    print(json.dumps(repair_check, ensure_ascii=False, indent=2))
    raise ValueError("Article failed source-grounded fact validation after repair.")
