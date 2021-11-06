import requests
import json
from collect_wikidata import WikidataUtility
import pandas as pd
import time

params = dict(
        action='wbsearchentities',
        format='json',
        language='en',
        uselang='en'
        )
url = 'https://www.wikidata.org/w/api.php?'


def phrase_to_ids(phrase):
    params["search"] = phrase
    response = requests.get(url, params).json()
    if "search" in response:
        response = response["search"]
    else:
        return None
    # print(json.dumps(response, indent=4))
    ids = [item["id"] for item in response]
    labels = [item["label"] for item in response]
    descs = [item.get("description", "no description") for item in response]
    records = list(zip(ids, labels, descs))
    print(*records, sep="\n")
    return ids, labels, descs


def initial_collect(phrase, outfile):
    wiki_util = WikidataUtility()
    ret = phrase_to_ids(phrase)
    if not ret:
        return
    else:
        ids, labels, descs = ret
    relations, objects, objectlabels = [], [], []
    for id, label in zip(ids, labels):
        wiki_util.query_subject(subject=id, subjectLabel=label, outfile=outfile)
        time.sleep(1.5)


with open("/home/thclark/stance/tropes/trope_results/semevalB_trope_uni_3000_20_200.txt") as f:
    tropes = f.read().split("\n")
    for trope in tropes:
        initial_collect(trope, "wikidata/from_tropes/trope_kg_semevalB.csv")
