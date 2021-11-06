# stance-knowledge
Stance Detection and Knowledge Infusion

## Stance-Relevant Knowledge Probing

See `notebooks/MaskedPrediction.ipynb` for an example of stance-relevant knowledge probing using the Masked Language Modeling mode of various BERT-based models. 

## Knowledge Infusion

### Knowledge Graph Format

We use SPARQL queries to collect knowledge triples from Wikidata. Each knowledge triple contains a subject, a relation, and an object. 

```
,subjectLabel,relation,objectLabel
0,theist,instance of,religious identity
1,theist,opposite of,atheist
2,theist,manifestation of,theism
3,ignosticism,instance of,world view
4,philosophy of religion,subclass of,philosophy
5,philosophy of religion,studies,religion
6,point of view,part of,psychology terminology
7,impious person,instance of,world view
8,impious person,instance of,religious identity
9,impious person,subclass of,person
10,religion,subclass of,world view
11,religion,studied by,theology
12,religion,studied by,philosophy of religion
13,philosophical theory,instance of,concept
14,philosophical theory,facet of,philosophy
15,atheism,instance of,world view
16,atheism,instance of,philosophical movement
17,atheism,instance of,doxastic attitude
18,atheism,subclass of,irreligion
19,atheism,opposite of,theism
20,atheism,dewey decimal classification,239.7
```

### Entity Enrichment

Entity Enrichment simply uses the SpaCy Entity Linker to look up entities within a text and identify their Wikidata ID and descriptions. These descriptions are appended as additional context to the text. 

### Knowledge Path Enrichment

Knowledge Path Enrichment identifies entities within a text and finds a path within the knowledge graph (if one exists) between in-text entities and the SD target. For example, if the SD target is Donald Trump and the text mentions Joe Biden, Knowledge Path Enrichment would find paths in the knowledge graph that connect Donald Trump and Joe Biden, such as the fact that they both held the office of President of the U.S. 

### Edge Cost Tuning

Edge Cost Tuning uses the training data to re-weight the edge costs of a knowledge graph to make Knowledge Path Enrichment more useful. Since there are often many possible paths in a knowledge graph between entities A and B, having good weights is important for selecting the most relevant knowledge path for a task. 
