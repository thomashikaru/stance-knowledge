import requests
import pandas as pd
import os
import time
import re
from graphs import Graph
import numpy as np


QUERY_URL = "https://query.wikidata.org/sparql"

COLUMNS = ["subject", "subjectLabel", "relation", "object", "objectLabel"]

TARGET_MAPPING = {
    "Donald Trump": "Q22686",
    "Hillary Clinton": "Q6294",
    "abortion": "Q8452",
    "climate change": "Q125928",
    "atheism": "Q7066",
    "feminism": "Q7252",
}


class WikidataUtility:
    """
    Handle SPARQL queries, reading & writing CSV files, etc.
    """

    def __init__(self):
        df = pd.read_json("props.json", orient="records")
        df = df[["label", "id"]]
        self.mapping = dict(zip(df["id"], df["label"]))

        self.OBJECT_QUERY = """
                SELECT 
                  ?subject ?subjectLabel ?relation
                WHERE {
                  {[] wikibase:directClaim ?relation}
                  ?subject ?relation wd:OBJECT .
                  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }
                """

        self.SUBJECT_QUERY = """
                SELECT 
                  ?object ?objectLabel ?relation
                WHERE {
                  {[] wikibase:directClaim ?relation}
                  wd:SUBJECT ?relation ?object .
                  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                }
                """

    def prop_id_to_label(self, prop_id):
        return self.mapping.get(prop_id, default=None)

    def query_object(self, obj, objectLabel, outfile):
        # build and send request, output result to temp file
        query = self.OBJECT_QUERY.replace("OBJECT", obj)
        r = requests.get(QUERY_URL, params={"format": "json", "query": query})
        if r.status_code != 200:
            print(r)
            return
        entries = list(r.json()["results"]["bindings"])
        # print(*entries, sep="\n")
        df = pd.DataFrame(entries)
        print(f"Number of Entries: {len(df)}")
        if len(df) == 0:
            return
        df["relation"] = df["relation"].apply(lambda x: x["value"].split("/")[-1])
        df["relation"] = df["relation"].replace(self.mapping)
        df["subject"] = df["subject"].apply(lambda x: x["value"].split("/")[-1])
        df["subjectLabel"] = df["subjectLabel"].apply(lambda x: x["value"])
        df["object"] = [obj] * len(df)
        df["objectLabel"] = [objectLabel] * len(df)
        if os.path.exists(outfile):
            old_df = pd.read_csv(outfile)
            df = df.append(old_df)
        df.drop_duplicates(inplace=True)
        df.to_csv(outfile, index=False, columns=COLUMNS)

    def query_subject(self, subject, subjectLabel, outfile):
        # build and send request, output result
        query = self.SUBJECT_QUERY.replace("SUBJECT", subject)
        r = requests.get(QUERY_URL, params={"format": "json", "query": query})
        if r.status_code != 200:
            print(r)
            return
        entries = list(r.json()["results"]["bindings"])
        # print(*entries, sep="\n")
        df = pd.DataFrame(entries)
        print(f"Number of Entries: {len(df)}")
        if len(df) == 0:
            return
        df["relation"] = df["relation"].apply(lambda x: x["value"].split("/")[-1])
        df["relation"] = df["relation"].replace(self.mapping)
        df["object"] = df["object"].apply(lambda x: x["value"].split("/")[-1])
        df["objectLabel"] = df["objectLabel"].apply(lambda x: x["value"])
        df["subject"] = [subject] * len(df)
        df["subjectLabel"] = [subjectLabel] * len(df)
        if os.path.exists(outfile):
            old_df = pd.read_csv(outfile)
            df = df.append(old_df)
        df.drop_duplicates(inplace=True)
        df.to_csv(outfile, index=False, columns=COLUMNS)


def test_graph():
    df = pd.read_csv("wikidata_results.csv")
    graph = Graph()
    for sub, obj in zip(df["subjectLabel"], df["objectLabel"]):
        graph.add_edge(sub, obj)
    graph.bfs("Donald Trump")


def collect_subject(filename, n, start_node):
    wiki_util = WikidataUtility()
    # wiki_util.query_subject(subject="Q22686", subjectLabel="Donald Trump", outfile="wikidata_results.csv")
    df = pd.read_csv(filename, dtype=str)

    # create graph of existing relationships
    graph = Graph()
    for sub, rel, obj in zip(df["subject"], df["relation"], df["object"]):
        graph.add_edge(sub, rel, obj)

    # get leaf objects (object entities with no outward edges) and corresponding labels
    leaf_objects = graph.bfs_leaves(start_node)
    print("Number of Leaf Objects:", len(df))
    pairs = dict(zip(df["object"], df["objectLabel"]))
    tuples = [(a, pairs[a]) for a in leaf_objects]

    # filter for only Wikidata Entity objects
    tuples = list(filter(lambda x: re.match("Q[0-9]+", str(x[0])), tuples))
    print(len(tuples))

    # collect new relationships with all leaf objects as subjects
    for subject, subjectLabel in tuples[: min(n, len(tuples))]:
        print(f"Collecting triples with subject of {subjectLabel}")
        wiki_util.query_subject(
            subject=subject, subjectLabel=subjectLabel, outfile=filename
        )
        time.sleep(1.5)


def collect_object(filename, n, start_node):
    wiki_util = WikidataUtility()
    df = pd.read_csv(filename)

    # create graph of existing relationships
    graph = Graph()
    for sub, rel, obj in zip(df["subject"], df["relation"], df["object"]):
        graph.add_edge(obj, rel, sub)

    # get leaf objects (object entities with no outward edges) and corresponding labels
    leaf_objects = graph.bfs_leaves(start_node)
    print("Number of Leaf Objects:", len(df))
    pairs = dict(zip(df["subject"], df["subjectLabel"]))
    tuples = [(a, pairs[a]) for a in leaf_objects]

    # filter for only Wikidata Entity objects
    tuples = list(filter(lambda x: re.match("Q[0-9]+", str(x[0])), tuples))
    print(len(tuples))

    # collect new relationships with all leaf objects as subjects
    for obj, objectLabel in tuples[: min(n, len(tuples))]:
        print(f"Collecting triples with object of {objectLabel}")
        wiki_util.query_object(obj=obj, objectLabel=objectLabel, outfile=filename)
        time.sleep(1.0)


def plot_graph(filename):
    df = pd.read_csv(filename)

    # create graph of existing relationships
    graph = Graph()
    for sub, obj in zip(df["subjectLabel"], df["objectLabel"]):
        graph.add_edge(sub, obj)

    graph.plot()


def main():
    # wiki_util = WikidataUtility()
    # wiki_util.query_subject(subject="Q7252", subjectLabel="feminism", outfile="wikidata/feminism/as_subject.csv")
    # wiki_util.query_object(obj="Q7252", objectLabel="feminism", outfile="wikidata/feminism/as_object.csv")
    #
    # wiki_util.query_subject(subject="Q7066", subjectLabel="atheism", outfile="wikidata/atheism/as_subject.csv")
    # wiki_util.query_object(obj="Q7066", objectLabel="atheism", outfile="wikidata/atheism/as_object.csv")
    #
    # wiki_util.query_subject(subject="Q125928", subjectLabel="climate change", outfile="wikidata/climate_change/as_subject.csv")
    # wiki_util.query_object(obj="Q125928", objectLabel="climate change", outfile="wikidata/climate_change/as_object.csv")

    collect_subject(
        filename="wikidata/feminism/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["feminism"],
    )

    collect_subject(
        filename="wikidata/atheism/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["atheism"],
    )

    collect_subject(
        filename="wikidata/climate_change/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["climate change"],
    )

    collect_subject(
        filename="wikidata/abortion/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["abortion"],
    )

    collect_subject(
        filename="wikidata/trump/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["Donald Trump"],
    )

    collect_subject(
        filename="wikidata/hillary_clinton/as_subject.csv",
        n=100,
        start_node=TARGET_MAPPING["Hillary Clinton"],
    )


if __name__ == "__main__":
    main()

