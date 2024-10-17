import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(G, bfs_order):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgray', font_weight='bold')

    # Highlight the BFS traversal path
    bfs_edges = [(bfs_order[i], bfs_order[i + 1]) for i in range(len(bfs_order) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=bfs_edges, edge_color='blue', width=2)

    plt.title("BFS Traversal")
    plt.show()

def BFS():
    visited_queue = []
    not_visited_queue = []
    G = nx.Graph()  # Initialize a new graph

    print("Enter your graph edges as 'node1 node2', type 'exit' when done:")
    while True:
        edge = input("Edge (or 'exit' to finish): ")
        if edge.lower() == 'exit':
            break
        else:
            node1, node2 = edge.split()
            G.add_edge(node1, node2)  # Add edges to the graph

    print("Graph edges:", G.edges())
    
    start_node = input("Enter the starting vertex for BFS: ")
    not_visited_queue.append(start_node)

    while not_visited_queue:
        current_node = not_visited_queue.pop(0)
        if current_node not in visited_queue:
            visited_queue.append(current_node)
            print("Visited:", current_node)

            # Add neighbors to the queue
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited_queue and neighbor not in not_visited_queue:
                    not_visited_queue.append(neighbor)

    print("BFS Order:", visited_queue)
    plot_graph(G, visited_queue)

BFS()
