import numpy as np
from gudhi import SimplexTree
from gudhi import RipsComplex
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Generate random points in R4 for tests
def generate_convex_set_r4(num_points=10):
    return np.random.rand(num_points, 4)

# Project points from R4 to R3: TODO
def project_to_r3(points_r4):
    return points_r4[:, :3]

# Compute the nerve complex using the Rips Complex (not sure if this is the best method?)
def compute_nerve(points_list, threshold=0.5):
    rips_complex = RipsComplex(points=points_list, max_edge_length=threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree

# Plot the convex hull in R3
def plot_convex_hull_r3(points_r3):
    hull = ConvexHull(points_r3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points_r3[:, 0], points_r3[:, 1], points_r3[:, 2], triangles=hull.simplices, alpha=0.3, edgecolor='k')
    plt.show()

# Construct nerve from graph G
def construct_nerve_from_graph(G, num_points=10, threshold=0.5):
    convex_sets = []
    for vertex in G:
        points_r4 = generate_convex_set_r4(num_points)
        points_r3 = project_to_r3(points_r4)
        convex_sets.append(points_r3)
        plot_convex_hull_r3(points_r3)  # Visualize each convex set in R3

    # Combine the sets for nerve computation
    points_combined = np.vstack(convex_sets)

    # Compute the nerve
    simplex_tree = compute_nerve(points_combined, threshold)
    return simplex_tree

# Example usage
# Define G (list of vertices or adjacency list)
G = [0, 1, 2, 3]  # simplified; TODO, construct a more meaningful example...

# Construct and visualize the nerve from G
nerve_complex = construct_nerve_from_graph(G)
print("Nerve complex simplices:", list(nerve_complex.get_skeleton(2)))


""" WIP
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import gudhi
import matplotlib.pyplot as plt
from itertools import combinations

G = nx.Graph()
G.add_edges_from([
    (0, 1),
    (1, 2),
    (2, 0)
])

def embed_graph(G, dim=4, tau=1.0):

    Embed the graph G into R^dim such that:
    - If two nodes are connected, their distance <= tau
    - Else, their distance > tau

    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Initial guess: random coordinates
    X0 = np.random.rand(n * dim)

    # Define the objective function
    def objective(X):
        X = X.reshape((n, dim))
        loss = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(X[i] - X[j])
                if G.has_edge(nodes[i], nodes[j]):
                    if dist > tau:
                        loss += (dist - tau)**2
                else:
                    if dist <= tau:
                        loss += (tau - dist)**2
        return loss

    # Optimize
    res = minimize(objective, X0, method='L-BFGS-B', options={'maxiter': 10000})

    if not res.success:
        raise ValueError("Failure: " + res.message)

    X = res.x.reshape((n, dim))
    return X, nodes

def construct_convex_sets(X, tau=1.0):

    Construct convex sets as balls in R^dim.
    Each ball is represented by its center and radius.

    radius = tau / 2
    convex_sets = [{'center': point, 'radius': radius} for point in X]
    return convex_sets

def verify_nerve(convex_sets, original_graph, epsilon=1e-3):

    Verify that the nerve of the convex sets matches the original graph.

    # Construct the nerve complex
    simplex_tree = gudhi.SimplexTree()

    n = len(convex_sets)
    for i in range(n):
        simplex_tree.insert([i])

    # Insert edges where convex sets intersect
    for i, j in combinations(range(n), 2):
        dist = np.linalg.norm(convex_sets[i]['center'] - convex_sets[j]['center'])
        if dist <= (convex_sets[i]['radius'] + convex_sets[j]['radius']) + epsilon:
            simplex_tree.insert([i, j])

    # Extract edges from the nerve
    nerve_graph = nx.Graph()
    for i in range(n):
        nerve_graph.add_node(i)

    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:
            nerve_graph.add_edge(simplex[0][0], simplex[0][1])

    # Compare with the original graph
    is_isomorphic = nx.is_isomorphic(original_graph, nerve_graph)
    print(f"Nerve matches the original graph: {is_isomorphic}")

    return is_isomorphic

def main():
    tau = 1.0  # threshold

    X, nodes = embed_graph(G, dim=4, tau=tau)
    print("Embedding coordinates:\n", X)

    convex_sets = construct_convex_sets(X, tau=tau)

    is_correct = verify_nerve(convex_sets, G)

    if is_correct:
        print("Success!")
    else:
        print("Failed to construct the nerve.")

    # Visualization (2D)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos={i: X_2d[j] for j, i in enumerate(nodes)}, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Graph Embedding Projection")
    plt.show()

if __name__ == "__main__":
    main()

"""
