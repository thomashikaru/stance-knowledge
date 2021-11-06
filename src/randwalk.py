import networkx as nx
import json
import random
import plotly.express as px
import argparse


def main(args):
    with open(args.input_file) as f:
        data = json.load(f)
    graph = nx.node_link_graph(data)

    for u, v, i in graph.edges:
        graph[u][v][i]["cost"] = 0

    N_ITERATIONS = 1000000
    TARGETS = args.targets.split(",")
    entities = list(graph.nodes)
    PATH_LENS = [int(x) for x in args.path_lens.split(",")]

    for i in range(N_ITERATIONS):
        t = random.choice(TARGETS)
        path_len = random.choice(PATH_LENS)
        for j in range(path_len):
            neighbors = list(graph.neighbors(t))
            choice = random.choice(neighbors)
            if j == path_len - 1:
                link = random.choice(list(graph[t][choice]))
                graph[t][choice][link]["cost"] -= 1
            t = choice

    # Pruning
    edges_to_remove = []
    for u, v, k in graph.edges:
        if graph[u][v][k]["cost"] == 0:
            edges_to_remove.append((u, v, k))
    graph.remove_edges_from(edges_to_remove)

    results = []
    for u, v, k in graph.edges:
        results.append(
            {
                "e1": u,
                "r": graph[u][v][k]["relation"],
                "e2": v,
                "cost": graph[u][v][k]["cost"],
            }
        )

    results = sorted(results, key=lambda x: x["cost"])
    print(*results[:20], sep="\n")
    print()
    print(*results[-10:], sep="\n")

    maxcost = abs(results[0]["cost"]) + 1
    print(maxcost)

    for u, v, i in graph.edges:
        graph[u][v][i]["cost"] += maxcost

    with open(args.output_file, "w") as f:
        json.dump(nx.node_link_data(graph), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", default="grimm_graph_randwalk.json")
    parser.add_argument(
        "output_file", default="grimm_graph_randwalk_weights_pruned.json"
    )
    parser.add_argument("n_iterations", type=int, default=1000000)
    parser.add_argument("targets", default="donald trump")
    parser.add_argument("path_lens", default="5,4")
    args = parser.parse_args()
    main(args)
