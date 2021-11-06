import spacy
from spacy_entity_linker import EntityLinker
import pandas as pd
from graphs import Graph
from descriptors import path_to_descriptions


def test_entity_linker():
    nlp = spacy.load("en_core_web_sm")
    entityLinker = EntityLinker(nlp, "entityLinker")
    nlp.add_pipe(entityLinker, last=True)

    doc = nlp(
        "Hillary can't use a fucking fax machine but you idiots are going to vote for her to be President?"
    )
    all_linked_entities = doc._.linkedEntities
    for ent in all_linked_entities:
        print(ent)
        print(ent.get_description())


def enrich():
    # entityLinker = EntityLinker()
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)

    # df = pd.read_csv("/home/thclark/stance/hashtag_generalization/hashtag_test.tsv", sep="\t")
    df = pd.read_csv(
        "/home/thclark/stance/semeval16/data-all-annotations/testdata-taskB-all-annotations.txt",
        sep="\t",
    )
    # df = pd.read_csv("/home/thclark/stance/grimminger/test.tsv", sep="\t")
    # df = df.sample(frac=1)

    mapping = {
        "Feminist Movement": "feminism",
        "Legalization of Abortion": "abortion",
        "Climate Change is a Real Concern": "climate change",
    }

    targets = df["Target"]
    targets = targets.replace(mapping)
    comments = df["Tweet"]

    # comments = df["cleaned_text"]
    # targets = ["donald trump"] * len(df)
    # comments = df["text"]

    descriptions = []

    count = 0
    for target, comment in zip(targets, comments):
        print(target)
        print(comment)
        doc = nlp(comment)
        desc_base = [f"Target: {target}."]
        result = []
        for ent in doc._.linkedEntities:
            result.append(f"{ent.get_label()}, {ent.get_description()}.")
        result = list(set(result))
        if len(result) > 3:
            result = result[:3]
        desc = " ".join(desc_base + result)
        print(desc, end="\n\n")
        descriptions.append(desc)
        count += 1

    # print(f"Enriched {count} / {len(comments)} comments, or {100*count//len(comments)}%.")
    df["Description"] = descriptions
    # df.to_csv("/home/thclark/stance/grimminger/test-enriched-stancy.tsv", sep="\t", index=False)
    df.to_csv(
        "/home/thclark/stance/semeval16/data-all-annotations/test-taskB-enriched-entities.tsv",
        sep="\t",
        index=False,
    )
    # df.to_csv("/home/thclark/stance/hashtag_generalization/hashtag_test_enriched.tsv", sep="\t", index=False)


def enrich_with_graph():
    graph = Graph()

    # graph.read_graph("semeval_graph_randwalk_weights_pruned.json")
    graph.read_graph("kg_semeval_weights_trained.json")

    # entityLinker = EntityLinker()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entityLinker", last=True)

    df = pd.read_csv(
        "/home/thclark/stance/semeval16/data-all-annotations/testdata-taskA-all-annotations.txt",
        sep="\t",
    )
    # df = pd.read_csv("/home/thclark/stance/semeval16/data-all-annotations/trainingdata-all-annotations.txt", sep="\t")
    # df = pd.read_csv("/home/thclark/stance/grimminger/train.tsv", sep="\t")
    # df = pd.read_csv("/home/thclark/stance/hashtag_generalization/hashtag_test.tsv", sep="\t")
    # df = df.sample(frac=1)

    mapping = {
        "Feminist Movement": "feminism",
        "Legalization of Abortion": "abortion",
        "Climate Change is a Real Concern": "climate change",
    }

    targets = df["Target"]
    targets = targets.replace(mapping)

    comments = df["Tweet"]

    # targets = ["Donald Trump"] * len(df)
    # # comments = df["cleaned_text"]
    # comments = df["text"]

    descriptions = []

    count = 0
    for target, comment in zip(targets, comments):
        print(target)
        print(comment)
        doc = nlp(comment)
        desc_base = [f"Target: {target}."]
        result = []

        # find paths
        for ent in doc._.linkedEntities:
            # print(ent.get_label())
            if not ent.get_label():
                continue
            if ent.get_label().lower() in graph.graph:
                path = graph.shortest_path(target.lower(), ent.get_label().lower())
                if path and len(path) < 4 and len(path) > 1:
                    result.append(" ".join(path_to_descriptions(path)))
            # result.append(f"{ent.get_label()}, {ent.get_description()}.")

        # if no paths, backoff to lookup
        if len(result) == 0:
            for ent in doc._.linkedEntities:
                result.append(f"{ent.get_label()}, {ent.get_description()}.")

        result = list(set(result))
        if len(result) > 3:
            result = result[:3]
        desc = " ".join(desc_base + result)
        print(desc, end="\n\n")
        descriptions.append(desc)
        count += 1

    # print(f"Enriched {count} / {len(comments)} comments, or {100*count//len(comments)}%.")
    df["Description"] = descriptions
    # df.to_csv("/home/thclark/stance/grimminger/train-enriched-paths-tuned.tsv", sep="\t", index=False)
    # df.to_csv("/home/thclark/stance/hashtag_generalization/hashtag_test_enriched_paths.tsv", sep="\t", index=False)
    df.to_csv(
        "/home/thclark/stance/semeval16/data-all-annotations/test-taskA-enriched-paths-tuned.tsv",
        sep="\t",
        index=False,
    )


def main():
    test_entity_linker()
    # enrich()
    # enrich_with_graph()


if __name__ == "__main__":
    main()
