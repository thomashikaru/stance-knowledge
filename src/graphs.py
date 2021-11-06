import json
import networkx as nx
import plotly.graph_objs as go
from typing import List
from collections import deque
import math
from fuzzywuzzy import fuzz
import pandas as pd
from stance_utils import load_df_and_filter


def match(s1: str, s2: str, fuzzy=True):
    if fuzzy:
        # return s1.lower() in s2.lower()
        # return re.match(f".*(^|\s){s1.lower()}($|\s).*", s2.lower())
        s1, s2 = s1.lower(), s2.lower()
        return fuzz.ratio(s1, s2) >= 85 and fuzz.ratio(s2, s1) >= 85
    else:
        return s1.lower() == s2.lower()


def no_cycles(path):
    ls = [obj for rel, obj in path]
    return len(ls) == len(set(ls))


# This class represents a directed graph using adjacency list representation
class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = nx.MultiDiGraph()

    def load(self, filenames):
        df = pd.DataFrame()
        for filename in filenames:
            df_temp = load_df_and_filter(filename)
            df = df.append(df_temp)
        df.drop_duplicates(inplace=True)

        # create graph of existing relationships
        for sub, rel, obj in zip(df["subjectLabel"], df["relation"], df["objectLabel"]):
            if sub != obj:
                self.add_edge(sub, rel, obj)
                self.add_edge(obj, rel + "_", sub)

    def get_nodes(self):
        return self.graph.nodes

    # function to add an edge to graph
    def add_edge(self, u, r, v):
        self.graph.add_edges_from([(u, v, {"relation": r})])

    def init_edge_costs(self):
        for u, v, i in self.graph.edges:
            self.graph[u][v][i]["cost"] = math.log(
                self.graph.out_degree[u] + self.graph.out_degree[v], 2
            )

    def init_edge_costs_log(self):
        for u, v, i in self.graph.edges:
            self.graph[u][v][i]["cost"] = math.log(self.graph[u][v][i]["cost"], 2)

    def update_costs(self, path, amount):
        for i, pair in enumerate(path[:-1]):
            u = pair[1]
            r = path[i + 1][0]
            v = path[i + 1][1]
            for idx in self.graph[u][v]:
                if self.graph[u][v][idx]["relation"] == r:
                    self.graph[u][v][idx]["cost"] += amount

    def update_costs_multiply(self, path, amount):
        for i, pair in enumerate(path[:-1]):
            u = pair[1]
            r = path[i + 1][0]
            v = path[i + 1][1]
            for idx in self.graph[u][v]:
                if self.graph[u][v][idx]["relation"] == r:
                    self.graph[u][v][idx]["cost"] *= amount

    def get_edge_cost(self, u, r, v):
        for idx in self.graph[u][v]:
            if self.graph[u][v][idx]["relation"] == r:
                return self.graph[u][v][idx]["cost"]

    # Function to print a BFS of graph
    def bfs(self, s):

        # Mark all the vertices as not visited
        entities = set(self.graph.nodes)
        visited = {x: False for x in entities}

        # Create a queue for BFS
        queue = [s]
        visited[s] = True

        while queue:
            s = queue.pop(0)
            print(s)
            for i in self.graph[s]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    # return leaves of a graph (nodes with no outward edges)
    def bfs_leaves(self, s):

        # Mark all the vertices as not visited
        entities = set(self.graph.nodes)
        return [x for x in entities if len(self.graph.successors(x)) == 0]

    # Utility function for printing
    # the found path in graph
    def printpath(self, path: List[int]) -> None:
        size = len(path)
        for a, b in path:
            print(f"({a}) {b}", end=" ")
        print()

    # Utility function for finding paths in graph
    # from source to destination
    def find_paths(self, src: str, dst: str, maxlen: int):

        # results
        results = []

        # Create a queue which stores the paths
        q = deque()

        # Path vector to store the current path
        path = [("Start", src.lower())]
        q.append(path.copy())

        while q:
            path = q.popleft()
            last = path[-1][1]

            # If last vertex is the desired destination
            # then print the path
            if (
                match(dst, last, fuzzy=True)
                and len(path) <= maxlen
                and not match(dst, src, fuzzy=True)
            ):
                results.append(path)
                continue  # stops searching a branch when dest. found

            # Traverse to all the nodes connected to
            # current vertex and push new path to queue
            if last in self.graph:
                for succ in self.graph.successors(last):
                    options = [
                        self.graph[last][succ][i] for i in self.graph[last][succ]
                    ]
                    for option in options:
                        item = (option["relation"], succ)
                        if item not in path and len(path) < maxlen and no_cycles(path):
                            newpath = path.copy()
                            newpath.append(item)
                            q.append(newpath)

        return results

    def shortest_path(self, src: str, dst: str):
        if src not in self.graph or dst not in self.graph:
            return None
        if not nx.has_path(self.graph, source=src, target=dst):
            return None
        path = nx.shortest_path(self.graph, source=src, target=dst, weight="cost")
        result = [("Start", src)]
        for u, v in zip(path[:-1], path[1:]):
            edge_data = list(self.graph.get_edge_data(u, v).items())
            result.append((edge_data[0][1]["relation"], v))
        return result

    def read_graph(self, inputfile):
        with open(inputfile) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def write_graph(self, outputfile):
        with open(outputfile, "w") as f:
            json.dump(nx.node_link_data(self.graph), f)

    def plot(self):
        """
        Plot a graph using networkx and Plotly.
        :return:
        """
        # add relation-labeled edges to a list and generate positions
        g = self.graph
        pos = nx.drawing.layout.spring_layout(g)

        edge_x, edge_y = [], []
        xtext, ytext = [], []
        etext = list(nx.get_edge_attributes(g, "relation").values())

        for edge in g.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            xtext.append((x0 + x1) / 2)
            ytext.append((y0 + y1) / 2)
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="text",
            mode="lines",
        )

        node_x, node_y = [], []
        for node in g.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            opacity=0.8,
            textposition="middle right",
            marker=dict(
                showscale=True,
                colorscale="burg",
                reversescale=False,
                color=[],
                size=[],
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=0,
            ),
        )

        relations_trace = go.Scatter(
            x=xtext,
            y=ytext,
            mode="text",
            marker_size=0.5,
            opacity=0.6,
            text=etext,
            textposition="top center",
            hovertemplate="relation: %{text}<extra></extra>",
        )

        node_adjacencies, node_text = [], []
        for node, adjacencies in enumerate(g.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        for node in enumerate(g.nodes()):
            node_text.append(node[1])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        node_trace.marker.size = [5 * x for x in node_adjacencies]

        fig = go.Figure(
            data=[edge_trace, node_trace, relations_trace],
            layout=go.Layout(
                title="<br>Knowledge Graph Connections",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        fig.show()


if __name__ == "__main__":
    pass
