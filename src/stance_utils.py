import pandas as pd
import re
import numpy as np

IGNORE_RELATIONS = [
    "Commons",
    "different from",
    "name in native language",
    "URL",
    "code",
    "website",
    "official name",
    "identifier",
    "ID",
    "native label",
    "inspired by",
    "fictional analog of",
    "diplomatic relation",
    "image",
    "Wiki",
    "described by source",
    "writing system",
    "on focus list of wikimedia project",
    "hascontext",
    "has properties for this type",
    "relatedto",
]

IGNORE_ENTITIES = [
    "human",
    "English",
    "male",
    "female",
    "Category:.*",
    "United States of America",
    "[0-9]+",
    "Template:.*",
    "Portal:.*",
    "http.*",
]


def load_df_and_filter(filename):
    df = pd.read_csv(
        filename,
        lineterminator="\n",
        dtype={"subjectLabel": str, "relation": str, "objectLabel": str},
    )
    df["subjectLabel"] = df["subjectLabel"].apply(
        lambda x: re.sub(r"_", r" ", str(x).lower())
    )
    df["objectLabel"] = df["objectLabel"].apply(
        lambda x: re.sub(r"_", r" ", str(x).lower())
    )
    df["relation"] = df["relation"].apply(lambda x: re.sub(r"_", r" ", str(x).lower()))
    mask = np.invert(
        df["relation"].str.contains("|".join(IGNORE_RELATIONS), case=False)
    )
    mask = np.bitwise_and(
        mask,
        np.invert(
            df["subjectLabel"].str.fullmatch(
                "|".join(IGNORE_ENTITIES), na=False, case=False
            )
        ),
    )
    mask = np.bitwise_and(
        mask,
        np.invert(
            df["objectLabel"].str.fullmatch(
                "|".join(IGNORE_ENTITIES), na=False, case=False
            )
        ),
    )
    df = df[mask]
    return df
