import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.manifold import MDS
from gudhi import SimplexTree
import networkx as nx
import warnings
from itertools import combinations
import time
import concurrent

"""
USAGE: python3.12 tda_final.py
Note: tested with 3.12 because there were setup problems with Gudhi when using 3.13.

OPTIONAL: Specify test cases to run by their numbers or ranges.
Examples: '1', '1-5', '10 15 20-25'.
USAGE: python3.12 tda_final.py -t 40-100.
If you type a test case greater than 60, the code will automatically generate
further examples with random regular graphs (see function below).
You can set the parameters yourself (number of nodes and graph degree)
by modifying the code below (where the add_random_regular_test_cases function is called).

OPTIONAL: pass the parameter -nv to disable visualization.
Note that visualization is available exclusively for planar graphs.
USAGE: python3.12 tda_final.py -t 40-100 -nv.

OPTIONAL: pass the parameter -a to abort the computation once the timeout has been reached.
Timeout: 600 seconds (10 minutes) by default. You can modify it yourself below and set it to a custom value, if you wish.
USAGE: python3.12 tda_final.py -t 40-100 -nv -a.
"""

def add_random_regular_test_cases(test_cases, start_n=5, d=4, num_cases=40, increment=5):
    """
    Adds Random Regular Graph from networkx test cases to the existing test_cases list.

    Parameters:
    - test_cases: Existing list of test case dictionaries.
    - start_n: Starting number of nodes (default: 5).
    - d: Degree for Random Regular Graphs (default: 4).
    - num_cases: Number of test cases to add (default: 40).
    - increment: Increment step for number of nodes (default: 5).
    """
    for i in range(num_cases):
        n = start_n + i * increment
        try:
            if (n * d) % 2 != 0:
                raise nx.NetworkXError(f"n*d must be even for a regular graph. Got n={n}, d={d}.")
            if d >= n:
                raise nx.NetworkXError(f"Degree d must be less than number of nodes n. Got n={n}, d={d}.")

            G = nx.random_regular_graph(d, n)
            edges = list(G.edges())

            test_case = {
                'title': f'Test Case {len(test_cases) + 1}: Random Regular Graph (n={n}, d={d})',
                'edges': edges,
                'multigraph': False
            }

            test_cases.append(test_case)

        except nx.NetworkXError as e:
            print(f"Failed to create Random Regular Graph with n={n}, d={d}: {e}")

def plot_cactus(node_counts, execution_times):
    """
    Creates and displays a cactus plot with the execution time vs. number of nodes.

    Parameters:
    - node_counts: List of node counts.
    - execution_times: List of execution times.
    """
    if node_counts and execution_times:
        plt.figure(figsize=(12, 8))
        plt.scatter(node_counts, execution_times, color='green', alpha=0.6, edgecolors='w', s=100)
        plt.title('Cactus Plot: Execution Time vs. Number of Nodes (Random Regular Graphs)')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cactus_plot_execution_time_vs_nodes.png')  # Save the plot as an image
        plt.show()
        print("\nCactus plot saved as 'cactus_plot_execution_time_vs_nodes.png'.")
    else:
        print("No data available to plot.")

warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step, clipping to bounds",
    category=RuntimeWarning
)

def random_tree(n, seed=None):
    """
    Generates a random tree with n nodes using the Prüfer sequence method.
    This is a hack because random_tree from networkx didn't work.

    Parameters:
    - n: Number of nodes in the tree.
    - seed: Seed for the random number generator.

    Returns:
    - edges: List of tuples with the edges of the tree.
    """
    import random
    if seed is not None:
        random.seed(seed)
    if n == 0:
        return []
    if n == 1:
        return []
    prufer = [random.randint(0, n - 1) for _ in range(n - 2)]
    node_degree = [1] * n
    for node in prufer:
        node_degree[node] += 1

    # the smallest leaf
    ptr = 0
    while ptr < n and node_degree[ptr] != 1:
        ptr += 1
    leaf = ptr

    edges = []
    for node in prufer:
        edges.append((leaf, node))
        node_degree[leaf] -= 1
        node_degree[node] -= 1
        if node_degree[node] == 1 and node < ptr:
            leaf = node
        else:
            ptr += 1
            while ptr < n and node_degree[ptr] != 1:
                ptr += 1
            leaf = ptr

    # the last edge
    u = -1
    v = -1
    for i in range(n):
        if node_degree[i] == 1:
            if u == -1:
                u = i
            else:
                v = i
                break
    if u != -1 and v != -1:
        edges.append((u, v))
    return edges

def edge_list_to_adjacency_list(edge_list, multigraph=False):
    """
    Converts an edge list to an adjacency list.

    Parameters:
    - edge_list: List of tuples representing edges.
    - multigraph: Boolean indicating if the graph is a multigraph.

    Returns:
    - adjacency: Dictionary representing the adjacency list.
    """
    if multigraph:
        G = nx.MultiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edge_list)
    adjacency = {node: set(neighbors) for node, neighbors in G.adjacency()}
    return adjacency

def is_planar_graph(G_edges):
    """
    Checks if a graph is planar.

    Parameters:
    - G_edges: List of tuples representing edges.

    Returns:
    - is_planar: Boolean indicating if the graph is planar.
    """
    G_nx = nx.Graph()
    G_nx.add_edges_from(G_edges)
    return nx.check_planarity(G_nx)[0]

def assign_positions_and_radii(G, initial_embedding_dim=6, final_embedding_dim=3, num_attempts=5):
    """
    Assigns positions and radii to graph vertices to satisfy overlap constraints.

    Parameters:
    - G: Adjacency list of the graph.
    - initial_embedding_dim: Initial dimension for embedding.
    - final_embedding_dim: Final dimension for embedding (used if graph is planar).
    - num_attempts: Number of optimization attempts.

    Returns:
    - convex_sets: Dictionary mapping vertices to their convex sets.
    - embedding_dim: The final embedding dimension used.
    """
    vertices = list(G.keys())
    n = len(vertices)

    adjacency = G

    min_radius = 0.1
    max_radius = 1.0
    epsilon = 1e-5
    delta = 1e-3    # min separation between non-overlapping balls

    # selected edge cases
    if n == 0:
        print("Empty Graph: No vertices to process.")
        return {}, initial_embedding_dim

    if n == 1:
        print("Single Vertex Graph: default position and radius.")
        convex_sets = {
            vertices[0]: {'center': np.zeros(initial_embedding_dim), 'radius': (max_radius + min_radius) / 2}
        }
        return convex_sets, initial_embedding_dim

    best_positions = None
    best_radii = None

    # MDS for initial positions
    # see here for more information: https://scikit-learn.org/dev/modules/generated/sklearn.manifold.MDS.html
    adjacency_matrix = np.zeros((n, n))
    for i, u in enumerate(vertices):
        for v in adjacency[u]:
            j = vertices.index(v)
            adjacency_matrix[i, j] = 1

    # dissimilarity: 0 if connected, 1 otherwise
    distances = np.where(adjacency_matrix == 1, 0, 1)
    mds = MDS(n_components=initial_embedding_dim, dissimilarity='precomputed', random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            positions_init = mds.fit_transform(distances)
        except ValueError as e:
            print(f"MDS failed: {e}")
            return None
    radii_init = np.full(n, (max_radius + min_radius) / 2)

    for attempt in range(num_attempts):
        # perturbation
        positions = positions_init + np.random.randn(n, initial_embedding_dim) * 0.1
        radii = radii_init + np.random.randn(n) * 0.05
        radii = np.clip(radii, min_radius, max_radius)

        x0 = np.hstack([positions.flatten(), radii])

        def objective(x):
            radii = x[n * initial_embedding_dim:]
            return np.sum(radii)

        constraints = []
        for idx_i in range(n):
            for idx_j in range(idx_i + 1, n):
                u = vertices[idx_i]
                v = vertices[idx_j]
                if v in adjacency[u]:
                    # Condition: balls must overlap (or touch)
                    def overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                        positions = x[:n * initial_embedding_dim].reshape((n, initial_embedding_dim))
                        radii = x[n * initial_embedding_dim:]
                        dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                        return radii[idx_i] + radii[idx_j] - dist - epsilon
                    constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                else:
                    # no overlap
                    def non_overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                        positions = x[:n * initial_embedding_dim].reshape((n, initial_embedding_dim))
                        radii = x[n * initial_embedding_dim:]
                        dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                        return dist - radii[idx_i] - radii[idx_j] - delta
                    constraints.append({'type': 'ineq', 'fun': non_overlap_constraint})

        bounds = [(-np.inf, np.inf)] * (n * initial_embedding_dim) + [(min_radius, max_radius)] * n

        res = minimize(
            objective,
            x0,
            method='SLSQP', # sequential least squares programming, see here: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 20000, 'ftol': 1e-12}
        )

        if res.success:
            x_opt = res.x
            positions_opt = x_opt[:n * initial_embedding_dim].reshape((n, initial_embedding_dim))
            radii_opt = x_opt[n * initial_embedding_dim:]

            # verify
            feasible = True
            for con in constraints:
                if con['type'] == 'ineq':
                    val = con['fun'](x_opt)
                    if val < -epsilon:
                        feasible = False
                        break
            if feasible:
                best_positions = positions_opt
                best_radii = radii_opt
                break  # Solution found!
        else:
            print(f"Attempt {attempt + 1}: Optimization failed. Trying again...")

    if best_positions is None:
        print("Optimization failure after multiple attempts...")
        return None

    # Project to final_embedding_dim? Use planarity.
    G_edges = [(u, v) for u in adjacency for v in adjacency[u] if u < v]
    is_planar = is_planar_graph(G_edges)

    if final_embedding_dim < initial_embedding_dim and is_planar:
        # lower dimension
        mds = MDS(n_components=final_embedding_dim, dissimilarity='euclidean', random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                positions_proj = mds.fit_transform(best_positions)
            except ValueError as e:
                print(f"MDS projection failed: {e}")
                return None
        embedding_dim = final_embedding_dim
    else:
        positions_proj = best_positions
        embedding_dim = initial_embedding_dim

    positions = positions_proj
    radii = best_radii
    x0 = np.hstack([positions.flatten(), radii])

    # new embedding
    constraints = []
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            u = vertices[idx_i]
            v = vertices[idx_j]
            if v in adjacency[u]:
                # Condition: balls must overlap (or touch)
                def overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                    positions = x[:n * embedding_dim].reshape((n, embedding_dim))
                    radii = x[n * embedding_dim:]
                    dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                    return radii[idx_i] + radii[idx_j] - dist - epsilon
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            else:
                # no overlap
                def non_overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                    positions = x[:n * embedding_dim].reshape((n, embedding_dim))
                    radii = x[n * embedding_dim:]
                    dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                    return dist - radii[idx_i] - radii[idx_j] - delta
                constraints.append({'type': 'ineq', 'fun': non_overlap_constraint})

    bounds = [(-np.inf, np.inf)] * (n * embedding_dim) + [(min_radius, max_radius)] * n

    res_refine = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 20000, 'ftol': 1e-12}
    )

    if res_refine.success:
        x_opt = res_refine.x
        positions_opt = x_opt[:n * embedding_dim].reshape((n, embedding_dim))
        radii_opt = x_opt[n * embedding_dim:]

        # verify
        feasible = True
        for con in constraints:
            if con['type'] == 'ineq':
                val = con['fun'](x_opt)
                if val < -epsilon:
                    feasible = False
                    break
        if not feasible:
            print("Problem: Constraints not satisfied.")
            return None
    else:
        print("Problem: Optimization failed in the final embedding dimension...")
        return None

    convex_sets = {
        vertices[i]: {'center': positions_opt[i], 'radius': radii_opt[i]}
        for i in range(n)
    }

    return convex_sets, embedding_dim

def compute_nerve_complex(convex_sets, epsilon=1e-6):
    """
    Computes the nerve complex from convex sets.

    Parameters:
    - convex_sets: Dictionary mapping vertices to their convex sets.
    - epsilon: Value required for numerical precision.

    Returns:
    - simplex_list: List of simplices in the nerve complex.
    """
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    if not all(isinstance(v, int) for v in vertices):
        vertex_to_int = {v: i for i, v in enumerate(vertices)}
    else:
        vertex_to_int = {v: v for v in vertices}
    int_to_vertex = {v_i: v for v, v_i in vertex_to_int.items()}

    # 0-simplices
    for v in vertices:
        simplex_tree.insert([vertex_to_int[v]])

    # 1-simplices if overlap found
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            v_i = vertices[i]
            v_j = vertices[j]
            set_i = convex_sets[v_i]
            set_j = convex_sets[v_j]
            distance = np.linalg.norm(set_i['center'] - set_j['center'])
            r_sum = set_i['radius'] + set_j['radius']
            if distance <= r_sum + epsilon:
                simplex_tree.insert([vertex_to_int[v_i], vertex_to_int[v_j]])

    simplex_list = []
    for simplex in simplex_tree.get_simplices():
        simplex_vertices = [int_to_vertex[v_int] for v_int in simplex[0]]
        simplex_list.append((simplex_vertices, simplex[1]))

    return simplex_list

def visualize_convex_sets(convex_sets, embedding_dim, title="Convex Sets"):
    """
    Visualizes convex sets in 3D space if the embedding dimension is 3.

    Parameters:
    - convex_sets: Dictionary mapping vertices to their convex sets.
    - embedding_dim: Dimension of the embedding.
    - title: Plot title.
    """
    if embedding_dim != 3:
        print(f"Visualization not available for embedding dimension {embedding_dim}")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(convex_sets)))

    for idx, (vertex, convex_set) in enumerate(convex_sets.items()):
        center = convex_set['center']
        radius = convex_set['radius']

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=colors[idx], alpha=0.3)
        ax.text(center[0], center[1], center[2], f'{vertex}', color='k')

    plt.title(title)
    plt.show()

def compare_nerve_and_graph(nerve_complex, G_edges):
    """
    Compares the nerve complex with the original graph to identify discrepancies.

    Parameters:
    - nerve_complex: List of simplices in the nerve complex.
    - G_edges: List of tuples representing edges in the original graph.

    Returns:
    - test_passed: Boolean indicating if the nerve complex matches the original graph.
    """
    nerve_edges = set()
    for simplex in nerve_complex:
        if len(simplex[0]) == 2:
            nerve_edges.add(tuple(sorted(simplex[0])))
    graph_edges = set(tuple(sorted(edge)) for edge in G_edges)
    missing_edges = graph_edges - nerve_edges
    extra_edges = nerve_edges - graph_edges
    discrepancies = False
    if missing_edges or extra_edges:
        discrepancies = True
        print("Discrepancies found:")
        if missing_edges:
            print("Edges missing in nerve:", missing_edges)
        if extra_edges:
            print("Extra edges in nerve:", extra_edges)
    else:
        print("Nerve complex matches the input graph :-)")
    return not discrepancies

def run_test_case(G_edges, title="Test Case", multigraph=False, visualize=True):
    """
    Executes a single test case.

    Parameters:
    - G_edges: List of tuples representing edges.
    - title: Title of the test case.
    - multigraph: Boolean indicating if the graph is a multigraph.
    - visualize: Boolean for including a visualization.

    Returns:
    - test_passed: Boolean indicating if the test passed.
    """
    G = edge_list_to_adjacency_list(G_edges, multigraph=multigraph)
    is_planar = is_planar_graph(G_edges)
    final_embedding_dim = 3 if is_planar else 6  # higher dimension for non-planar graphs
    result = assign_positions_and_radii(
        G, initial_embedding_dim=6, final_embedding_dim=final_embedding_dim, num_attempts=5
    )

    if result is not None:
        convex_sets, embedding_dim = result
        nerve_complex = compute_nerve_complex(convex_sets)
        if visualize:
            visualize_convex_sets(convex_sets, embedding_dim, title=title)
        test_passed = compare_nerve_and_graph(nerve_complex, G_edges)

        if not convex_sets:
            print("No convex sets were constructed (Empty Graph). Test Passed by Default.")
            test_passed = True

        print("Nerve complex simplices:")
        for simplex in sorted(nerve_complex, key=lambda s: (len(s[0]), s[0])):
            print(simplex)
    else:
        print("Failed to construct convex sets for", title)
        test_passed = False

    return test_passed

def run_test_case_with_timeout(edges, title, multigraph, visualize):
    """
    Wrapper function to run a test case and measure its execution time.

    Parameters:
    - edges: List of tuples representing edges.
    - title: Title of the test case.
    - multigraph: Boolean indicating if the graph is a multigraph.
    - visualize: Boolean indicating whether to visualize the convex sets.

    Returns:
    - Tuple (test_passed, duration)
    """
    start_time = time.perf_counter()
    test_passed = run_test_case(edges, title, multigraph, visualize)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return test_passed, duration

def main():
    """
    Main function to parse CL arguments and run test cases.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # 60 hardcoded test cases
    test_cases = [
        {
            'title': 'Test Case 1: Simple Triangle Graph',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 0)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 2: Star Graph',
            'edges': [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 3: Complete Graph K4',
            'edges': [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 4: Linear Chain',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 5: Kuratowski K3,3 Graph',
            'edges': [
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 6: Tree Graph',
            'edges': [
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (2, 6)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 7: Cycle with a Chord',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 2)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 8: Disconnected Graph',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 9: Petersen Graph',
            'edges': list(nx.petersen_graph().edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 10: 3D Hypercube Graph',
            'edges': list(nx.convert_node_labels_to_integers(nx.hypercube_graph(3)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 11: Complete Bipartite Graph K3,3',
            'edges': list(nx.complete_bipartite_graph(3, 3).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 12: Wheel Graph W6',
            'edges': list(nx.wheel_graph(6).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 13: Ladder Graph',
            'edges': list(nx.ladder_graph(2).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 14: Grid Graph 3x3',
            'edges': list(nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 15: Barbell Graph',
            'edges': list(nx.barbell_graph(3, 0).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 16: Path Graph P5',
            'edges': list(nx.path_graph(5).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 17: Cycle Graph C6',
            'edges': list(nx.cycle_graph(6).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 18: Friendship Graph with 5 triangles',
            'edges': list(nx.convert_node_labels_to_integers(nx.windmill_graph(5, 3)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 19: Random Geometric Graph G(20, r=0.3)',
            'edges': [],  # construction below
            'multigraph': False
        },
        {
            'title': 'Test Case 20: 4D Hypercube Graph',
            'edges': list(nx.convert_node_labels_to_integers(nx.hypercube_graph(4)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 21: Large Sparse Random Graph',
            'edges': list(nx.gnp_random_graph(30, 0.05).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 22: Dense Random Graph',
            'edges': list(nx.gnp_random_graph(20, 0.5).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 23: Binary Tree of Depth 4',
            'edges': list(nx.balanced_tree(r=2, h=4).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 24: Complete Graph K5',
            'edges': list(nx.complete_graph(5).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 25: Cycle Graph with 10 Nodes',
            'edges': list(nx.cycle_graph(10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 26: Line Graph with 15 Nodes',
            'edges': list(nx.path_graph(15).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 27: Star Graph with 10 Leaves',
            'edges': list(nx.star_graph(10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 28: Disconnected Graph with Multiple Components',
            'edges': list(nx.Graph([
                (0, 1), (1, 2), (2, 0),
                (3, 4), (4, 5), (5, 3),
                (6, 7), (7, 8), (8, 6)
            ]).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 29: Bipartite Graph K3,5',
            'edges': list(nx.complete_bipartite_graph(3, 5).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 30: Planar Grid Graph 5x5',
            'edges': list(nx.convert_node_labels_to_integers(nx.grid_2d_graph(5, 5)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 31: Grötzsch Graph',
            'edges': list(nx.mycielski_graph(4).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 32: Kneser Graph K5,2',
            'edges': [],
            'multigraph': False
        },
        {
            'title': 'Test Case 33: Graph with Constraints',
            'edges': [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 4), (3, 5), (3, 6),
                (4, 5), (5, 6),
                (0, 5), (0, 6)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 34: Empty Graph',
            'edges': [],
            'multigraph': False
        },
        {
            'title': 'Test Case 35: Random Tree with 10 Nodes',
            'edges': [], # see below
            'multigraph': False
        },
        {
            'title': 'Test Case 36: Random Tree with 27 Nodes',
            'edges': [], # see below
            'multigraph': False
        },
        {
            'title': 'Test Case 37: Two Nodes with an Edge',
            'edges': [
                (0, 1)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 38: Two Triangles Connected by a Single Edge',
            'edges': [

                (0, 1),
                (1, 2),
                (2, 0),

                (3, 4),
                (4, 5),
                (5, 3),

                (2, 3)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 39: Multigraph with Parallel Edges',
            'edges': [
                (0, 1),
                (0, 1),
                (1, 2),
                (1, 2),
                (2, 0),
                (2, 0)
            ],
            'multigraph': True
        },
        {
            'title': 'Test Case 40: Large Complete Graph K10',
            'edges': list(nx.complete_graph(10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 41: Outerplanar Graph',
            'edges': [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (0, 2),
                (1, 3)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 42: Series-Parallel Graph',
            'edges': [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                (1, 3), (3, 5)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 43: Chordal Graph',
            'edges': [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (0, 2),
                (1, 3),
                (2, 4), (3, 4)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 44: Highly Asymmetric Tree',
            'edges': [
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 5), (5, 6), (6, 7)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 45: Graph with High Diameter',
            'edges': list(nx.path_graph(20).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 46: Torus Graph',
            'edges': [],  # see below
            'multigraph': False
        },
        {
            'title': 'Test Case 47: Random Directed Graph',
            'edges': list(nx.gn_graph(10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 48: 3-Regular Graph',
            'edges': list(nx.random_regular_graph(3, 10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 49: Large Random Tree (50 Nodes)',
            'edges': [], # construction below
            'multigraph': False
        },
        {
            'title': 'Test Case 50: Barbell Graph with Long Paths',
            'edges': list(nx.barbell_graph(5, 10).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 51: Erdos-Renyi Graph G(n, p=0.1)',
            'edges': list(nx.erdos_renyi_graph(30, 0.1).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 52: Erdos-Renyi Graph G(n, p=0.9)',
            'edges': list(nx.erdos_renyi_graph(30, 0.9).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 53: Hexagonal Lattice Graph',
            'edges': list(nx.hexagonal_lattice_graph(3, 3).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 54: Line Graph of Complete Graph K5',
            'edges': list(nx.line_graph(nx.complete_graph(5)).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 55: Large Sparse Random Graph (100 Nodes)',
            'edges': list(nx.gnm_random_graph(100, 200).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 56: Large Dense Random Graph (100 Nodes)',
            'edges': list(nx.gnm_random_graph(100, 4950).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 57: Large Dense Random Graph (70 Nodes)',
            'edges': list(nx.gnm_random_graph(70, 5000).edges()),
            'multigraph': False
        },
        {
            'title': 'Test Case 58: Graph with Bridges',
            'edges': [
                (0, 1), (1, 2), (2, 3),
                (2, 4), (4, 5),
                (5, 6), (6, 7),
                (3, 7)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 59: Graph with 3 Cliques',
            'edges': [

                (0, 1), (1, 2), (2, 0),

                (2, 3), (3, 4), (4, 2),

                (4, 5), (5, 6), (6, 4),

                (1, 3), (3, 5)
            ],
            'multigraph': False
        },
        {
            'title': 'Test Case 60: Graph with a High Clustering Coefficient',
            'edges': [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3),
                (2, 3),
                (3, 4), (4, 5), (5, 3),
                (4, 6), (6, 5)
            ],
            'multigraph': False
        }
    ]

    add_random_regular_test_cases(test_cases, start_n=5, d=8, num_cases=40, increment=5)


    # Test Case 19: Random Geometric Graph G(20, r=0.3)
    np.random.seed(20)
    positions_19 = np.random.rand(20, 2)
    r_19 = 0.3
    edges_19 = []
    for i in range(len(positions_19)):
        for j in range(i + 1, len(positions_19)):
            if np.linalg.norm(positions_19[i] - positions_19[j]) < r_19:
                edges_19.append((i, j))
    test_cases[18]['edges'] = edges_19

    # Test Case 32: Kneser Graph K5,2
    subsets = list(combinations(range(5), 2))
    num_vertices = len(subsets)
    edges_kneser = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if set(subsets[i]).isdisjoint(subsets[j]):
                edges_kneser.append((i, j))
    test_cases[31]['edges'] = edges_kneser

    # Test Case 46: Torus Graph
    G_torus = nx.cartesian_product(nx.cycle_graph(3), nx.cycle_graph(3))
    edges_torus = list(G_torus.edges())
    test_cases[45]['edges'] = edges_torus

    # Test Case 49: Large Random Tree (50 Nodes)
    G_random_tree_49 = random_tree(50, seed=42)
    test_cases[48]['edges'] = G_random_tree_49

    # Test Case 35: Random Tree (10 Nodes)
    G_random_tree_35 = random_tree(10, seed=24)
    test_cases[34]['edges'] = G_random_tree_35

    # Test Case 36: Random Tree (27 Nodes)
    G_random_tree_36 = random_tree(27, seed=59)
    test_cases[35]['edges'] = G_random_tree_36

    for idx, test_case in enumerate(test_cases, start=1):
        test_case['number'] = idx


    parser = argparse.ArgumentParser(
        description="Run selected test cases."
    )
    parser.add_argument(
        '-t', '--tests',
        nargs='+',
        help=(
            "Specify test cases to run by their numbers or ranges. "
            "Examples: '1', '1-5', '10 15 20-25'"
        )
    )

    parser.add_argument(
        '-nv', '--no-visualization',
        action='store_true',
        help="Disable visualization of convex sets."
    )

    parser.add_argument(
        '-a', '--abort',
        action='store_true',
        help="Abort a test case if execution time exceeds 600 seconds."
    )

    args = parser.parse_args()

    abort = args.abort

    visualize = not args.no_visualization

    def parse_test_cases(selection, max_test):
        selected = set()
        for part in selection:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start = int(start)
                    end = int(end)
                    if start > end or start < 1 or end > max_test:
                        raise ValueError
                    selected.update(range(start, end + 1))
                except ValueError:
                    print(f"Invalid range specification: '{part}'. Skipping.")
            else:
                try:
                    num = int(part)
                    if num < 1 or num > max_test:
                        raise ValueError
                    selected.add(num)
                except ValueError:
                    print(f"Invalid test case number: '{part}'. Skipping.")
        return sorted(selected)

    if args.tests:
        selected_numbers = parse_test_cases(args.tests, len(test_cases))
        if not selected_numbers:
            print("No valid test cases selected. Exiting.")
            return
        selected_test_cases = [tc for tc in test_cases if tc['number'] in selected_numbers]
    else:
        # run all tests
        selected_test_cases = test_cases

    total_tests = len(selected_test_cases)
    passed_tests = 0
    failed_tests = 0
    failed_test_cases = []
    node_counts = []
    timings = {}
    execution_times = []
    total_time = 0

    if abort:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    for test_case in selected_test_cases:
        print("\n" + "="*50)
        print(f"{test_case['title']}")
        print("="*50)

        num_nodes = len(set([node for edge in test_case['edges'] for node in edge]))

        if abort:
            future = executor.submit(
                run_test_case_with_timeout,
                test_case['edges'],
                test_case['title'],
                test_case['multigraph'],
                visualize=visualize
            )
            try:
                test_passed, duration = future.result(timeout=600)
                print(f"Execution Time: {duration:.2f} seconds")
            except concurrent.futures.TimeoutError:
                print(f"Test case '{test_case['title']}' exceeded 600 seconds and was aborted.")
                test_passed = False
                duration = 600.0
        else:
            start_time = time.perf_counter()
            test_passed = run_test_case(
                test_case['edges'],
                title=test_case['title'],
                multigraph=test_case['multigraph'],
                visualize=visualize
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"Execution Time: {duration:.2f} seconds")

        timings[test_case['title']] = duration
        total_time += duration

        # plot data
        if 'Random Regular Graph' in test_case['title']:
            node_counts.append(num_nodes)
            execution_times.append(duration)

        if test_passed:
            passed_tests += 1
        else:
            failed_tests += 1
            failed_test_cases.append(test_case['title'])

    if abort:
        executor.shutdown(wait=False)

    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {failed_tests}")
    print(f"Total Execution Time: {total_time:.4f} seconds")

    if total_tests > 0:
        average_time = total_time / total_tests
        print(f"Average Time per Test: {average_time:.4f} seconds")
    else:
        print("No tests were run.")

    if failed_tests > 0:
        print("\nFailed Test Cases:")
        for title in failed_test_cases:
            print(f"- {title}")

    print("\n" + "="*50)
    print("Detailed Execution Times")
    print("="*50)
    for title, duration in timings.items():
        print(f"{title}: {duration:.4f} seconds")

    plot_cactus(node_counts, execution_times)

if __name__ == "__main__":
    main()
