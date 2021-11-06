from graphs import Graph
from stance_utils import load_df_and_filter
import pandas as pd
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from nltk import ngrams
import nltk
from nltk.corpus import stopwords
from typing import List, Tuple


mapping_wikidata = {
    "subclass of": "{sub} is a subclass of {obj}",
    "facet of": "{sub} is a facet of {obj}",
    "followed by": "{sub} was followed by {obj}",
    "follows": "{sub} follows {obj}",
    "participant in": "{sub} was a participant in {obj}",
    "immediate cause of": "{sub} was an immediate cause of {obj}",
    "position held": "{sub} held the position of {obj}",
    "part of": "{sub} is a part of {obj}",
    "office contested": "{sub} contested the office of {obj}",
    "educated at": "{sub} was educated at {obj}",
    "partially coincident with": "{sub} partially coincided with {obj}",
    "candidacy in election": "{sub} was a candidate in the {obj}",
    "significant event": "a significant event for {sub} was {obj}",
    "has cause": "{obj} is the cause of {sub}",
    "has organizer": "{obj} is the organizer of {sub}",
    "convicted of": "{sub} was convicted of {obj}",
    "participant": "{obj} was a participant in {sub}",
    "drug used for treatment": "{obj} is a drug used for treating {sub}",
    "owner of": "{sub} is the owner of {obj}",
    "replaces": "{sub} replaces {obj}",
    "item disputed by": "{sub} is disputed by {obj}",
    "depicts": "{sub} depicts {obj}",
    "founded by": "{sub} was founded by {obj}",
    "named after": "{sub} is named after {obj}",
    "in opposition to": "{sub} opposes {obj}",
    "appointed by": "{sub} is appointed by {obj}",
    "opposite of": "{sub} is the opposite of {obj}",
    "instance of": "{sub} is an instance of {obj}",
}

mapping_conceptnet = {
    "hascontext": "{sub} has context {obj}",
    "relatedto": "{sub} is related to {obj}",
    "derivedfrom": "{sub} is derived from {obj}",
    "isa": "{sub} is a kind of {obj}",
    "formof": "{sub} is a form of {obj}",
    "notdesires": "{sub} does not desire {obj}",
    "capableof": "{sub} is capable of {obj}",
    "causesdesire": "{sub} causes a desire for {obj}",
    "usedfor": "{sub} is used for {obj}",
    "etymologicallyrelatedto": "{sub} is etymologically related to {obj}",
    "definedas": "{sub} is defined as {obj}",
    "atlocation": "{sub} is located in {obj}",
}

mapping = {**mapping_conceptnet, **mapping_wikidata}


def triple_to_description(sub: str, rel: str, obj: str):
    """convert a KG triple to a pseudo-natural language description.

    Args:
        sub (str): subject
        rel (str): relation
        obj (str): object

    Returns:
        str: description
    """
    if rel.endswith("_"):
        rel = rel[:-1]
        sub, obj = obj, sub
    desc = f"{sub} has {rel} {obj}"
    if rel.startswith("has"):
        desc = f"{sub} {rel} {obj}"
    if rel in mapping:
        desc = mapping[rel].format(sub=sub, obj=obj)
    desc = desc[0] + desc[1:] + "."
    return desc


def path_to_descriptions(path: List[Tuple[str]]) -> List[str]:
    """convert a KG path to a list of descriptions.

    Args:
        path (List[Tuple[str]]): list of KG triples forming a path

    Returns:
        List[str]: list of descriptions corresponding to this path
    """
    results = []
    prev = None
    for i, (rel, obj) in enumerate(path):
        if i != 0:
            desc = triple_to_description(prev, rel, obj)
            results.append(desc)
        prev = obj
    return results


def path_to_descriptions_plus(path: Tuple[str], graph: object):
    """convet path to list of descriptions, returning product of path lengths as well

    Args:
        path (Tuple[str]): path in KG
        graph (object): KG

    Returns:
        Tuple: product of degrees, list of descriptions
    """
    results = []
    prev = None
    product = 1
    for i, (rel, obj) in enumerate(path):
        if i != 0:
            desc = triple_to_description(prev, rel, obj)
            product *= len(graph.graph[obj])
            results.append(desc)
        prev = obj
    return product, results


def path_to_descriptions_plus_nx(path, graph):
    results = []
    prev = None
    sum = 0
    for i, (rel, obj) in enumerate(path):
        if i != 0:
            desc = triple_to_description(prev, rel, obj)
            sum += graph.get_edge_cost(prev, rel, obj)
            results.append(desc)
        prev = obj
    return sum, results


class Enricher:
    def __init__(self, filenames, tropes_file=None):
        # read in knowledge graph
        df = pd.DataFrame()
        for filename in filenames:
            df_temp = load_df_and_filter(filename)
            df = df.append(df_temp)
        df.drop_duplicates(inplace=True)

        # handles tropes file, if provided
        if tropes_file:
            with open(tropes_file) as f:
                self.tropes = f.read().split("\n")
        else:
            self.tropes = None

        # create graph of existing relationships
        self.graph = Graph()
        for sub, rel, obj in zip(df["subjectLabel"], df["relation"], df["objectLabel"]):
            if sub != obj:
                self.graph.add_edge(sub, rel, obj)
                self.graph.add_edge(obj, rel + "_", sub)

        self.graph.init_edge_costs()

        self.entities = [x.lower() for x in self.graph.get_nodes()]
        # self.entities = [x.lower() for x in self.graph.graph.keys()]
        self.entities = set(self.entities)

        nltk.download("stopwords")
        self.stopwords = set(stopwords.words("english"))

        # stanza.download("en")
        # self.nlp = stanza.Pipeline("en", processors="tokenize,ner")

    def comment_to_entities_ner(self, comment):
        """get entities from a comment using named entity recognition

        Args:
            comment (str): a comment/tweet/sentence

        Returns:
            List[str]: list of entities
        """
        result = []
        sentences = self.nlp(comment).sentences
        for sent in sentences:
            result += sent.ents
        # print(result)
        result = [x.text for x in result]
        return result

    def comment_to_entities_fuzz(self, comment):
        """get entities from a comment using fuzzy string matching

        Args:
            comment (str): a comment/tweet/sentence

        Returns:
            List[str]: list of entities
        """
        if self.tropes is None:
            print("self.tropes is not defined")
            return
        bigrams = ngrams(comment.split(), n=2)
        results = []
        for bigram in bigrams:
            # print(bigram)
            bigram = " ".join(bigram)
            match = process.extractOne(bigram, self.tropes)
            if match[1] >= 80 and fuzz.ratio(match, bigram) >= 80:
                results.append(match)
        # results = list(sorted(filter(lambda x: x[1] > 86, set(results)), key=lambda x: x[1]))
        # print(results[-5:])
        # print(results)
        results = [a for a, b in results]
        return results

    def comment_to_entities_kg(self, comment):
        comment_words = [x for x in comment.lower().split() if x not in self.stopwords]
        bigrams = ngrams(comment_words, n=2)
        results = []
        for bigram in bigrams:
            bigram = " ".join(bigram)
            # match = process.extractOne(bigram, self.entities)
            # if match[1] >= 80 and fuzz.ratio(match, bigram) >= 80:
            #     results.append(match)
            if bigram in self.entities:
                results.append(bigram)
        for word in comment_words:
            if word in self.entities:
                results.append(word)
        # print(results)
        return results

    def comment_to_description(self, target, comment, maxlen):
        results = []

        if self.tropes is None:
            comment_to_entities = self.comment_to_entities_ner
        else:
            comment_to_entities = self.comment_to_entities_fuzz
        comment_to_entities = self.comment_to_entities_kg

        for dst in comment_to_entities(comment):
            paths = self.graph.find_paths(target, dst, maxlen=maxlen)
            if paths:
                results.append(min(paths, key=lambda x: x[0]))

        if not results:
            return None
        descriptions = [path_to_descriptions_plus(x, self.graph) for x in results]
        descriptions = list(sorted(descriptions, key=lambda x: x[0]))
        return descriptions

    def comment_to_response(self, target, comment, maxlen):
        results = []
        for dst in self.comment_to_entities_kg(comment):
            paths = self.graph.find_paths(target, dst, maxlen=maxlen)
            if paths:
                results.append(min(paths, key=lambda x: x[0]))
        if not results:
            return None

        response = []
        for x in results:
            sum, descriptions = path_to_descriptions_plus_nx(x, self.graph)
            response.append(
                {"path": x, "score": sum, "description": " ".join(descriptions)}
            )
        response = list(sorted(response, key=lambda x: x["score"]))
        return response

    def comment_to_response_shortest(self, target, comment, maxlen):
        results = []
        for dst in self.comment_to_entities_kg(comment):
            path = self.graph.shortest_path(target, dst)
            if path and len(path) <= maxlen:
                results.append(path)
        if not results:
            return None

        response = []
        for x in results:
            sum, descriptions = path_to_descriptions_plus_nx(x, self.graph)
            response.append(
                {"path": x, "score": sum, "description": " ".join(descriptions)}
            )
        response = list(sorted(response, key=lambda x: x["score"]))
        return response


if __name__ == "__main__":
    pass
