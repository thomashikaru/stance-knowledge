from graphs import Graph
import argparse


def main(args):
    graph = Graph()
    graph.read_graph(args.graph_file)
    for u, v, i in graph.graph.edges:
        if graph.graph[u][v][i]["cost"] < 0:
            print(u, v)
    print(graph.shortest_path(args.e1, args.e2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_file", default="semeval_graph_randwalk_weights_pruned.json"
    )
    parser.add_argument("e1", default="climate change")
    parser.add_argument("e2", default="carbon")
    args = parser.parse_args()
    main(args)
