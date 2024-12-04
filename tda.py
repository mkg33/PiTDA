import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.manifold import MDS
from gudhi import SimplexTree
import networkx as nx
import warnings

def edge_list_to_adjacency_list(edge_list):
    adjacency = {}
    for u, v in edge_list:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency

def is_planar_graph(G_edges):
    G_nx = nx.Graph()
    G_nx.add_edges_from(G_edges)
    return nx.check_planarity(G_nx)[0]

def assign_positions_and_radii(G, initial_embedding_dim=6, final_embedding_dim=3, num_attempts=5):
    vertices = list(G.keys())
    n = len(vertices)

    best_positions = None
    best_radii = None

    adjacency = G

    min_radius = 0.1
    max_radius = 1.0
    epsilon = 1e-5  # For numerical precision
    delta = 1e-3    # Minimum separation between non-overlapping balls

    # Use MDS for initial positions
    adjacency_matrix = np.zeros((n, n))
    for i, u in enumerate(vertices):
        for v in adjacency[u]:
            j = vertices.index(v)
            adjacency_matrix[i, j] = 1

    distances = np.where(adjacency_matrix == 1, 0, 1)
    mds = MDS(n_components=initial_embedding_dim, dissimilarity='precomputed', random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        positions_init = mds.fit_transform(distances)
    radii_init = np.full(n, (max_radius + min_radius) / 2)

    for attempt in range(num_attempts):
        # initial perturbation
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
                    # balls must overlap or touch
                    def overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                        positions = x[:n * initial_embedding_dim].reshape((n, initial_embedding_dim))
                        radii = x[n * initial_embedding_dim:]
                        dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                        return radii[idx_i] + radii[idx_j] - dist - epsilon
                    constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                else:
                    # balls must not overlap
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
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 20000, 'ftol': 1e-12}
        )

        if res.success:
            x_opt = res.x
            positions_opt = x_opt[:n * initial_embedding_dim].reshape((n, initial_embedding_dim))
            radii_opt = x_opt[n * initial_embedding_dim:]

            # verification
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
                break  # solution found!
        else:
            print(f"Attempt {attempt + 1}: Optimization failed. Trying again...")

    if best_positions is None:
        print("Optimization failure after multiple attempts...")
        return None

    # use planarity to decide whether to project to final_embedding_dim
    G_edges = [(u, v) for u in adjacency for v in adjacency[u] if u < v]
    is_planar = is_planar_graph(G_edges)

    if final_embedding_dim < initial_embedding_dim and is_planar:
        # project to lower dimension via MDS
        mds = MDS(n_components=final_embedding_dim, dissimilarity='euclidean', random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            positions_proj = mds.fit_transform(best_positions)
        embedding_dim = final_embedding_dim
    else:
        positions_proj = best_positions
        embedding_dim = initial_embedding_dim

    positions = positions_proj
    radii = best_radii
    x0 = np.hstack([positions.flatten(), radii])

    # Redefine constraints for the new embedding
    constraints = []
    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            u = vertices[idx_i]
            v = vertices[idx_j]
            if v in adjacency[u]:
                # balls must overlap or touch
                def overlap_constraint(x, idx_i=idx_i, idx_j=idx_j):
                    positions = x[:n * embedding_dim].reshape((n, embedding_dim))
                    radii = x[n * embedding_dim:]
                    dist = np.linalg.norm(positions[idx_i] - positions[idx_j])
                    return radii[idx_i] + radii[idx_j] - dist - epsilon
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
            else:
                # balls must not overlap
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

        # verification
        feasible = True
        for con in constraints:
            if con['type'] == 'ineq':
                val = con['fun'](x_opt)
                if val < -epsilon:
                    feasible = False
                    break
        if not feasible:
            print("Problem: Constraints not satisfied after refinement.")
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
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    # vertex labels as integers
    if not all(isinstance(v, int) for v in vertices):
        vertex_to_int = {v: i for i, v in enumerate(vertices)}
    else:
        vertex_to_int = {v: v for v in vertices}
    int_to_vertex = {v_i: v for v, v_i in vertex_to_int.items()}

    # 0-simplices
    for v in vertices:
        simplex_tree.insert([vertex_to_int[v]])

    # insert 1-simplices if overlap found
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

    # label mapping
    simplex_list = []
    for simplex in simplex_tree.get_simplices():
        simplex_vertices = [int_to_vertex[v_int] for v_int in simplex[0]]
        simplex_list.append((simplex_vertices, simplex[1]))

    return simplex_list

def visualize_convex_sets(convex_sets, embedding_dim, title="Convex Sets"):
    if embedding_dim != 3:
        print(f"Visualization not available for embedding dimension {embedding_dim}")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(convex_sets)))

    for idx, (vertex, convex_set) in enumerate(convex_sets.items()):
        center = convex_set['center']
        radius = convex_set['radius']

        # spherical coordinates
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=colors[idx], alpha=0.3)
        ax.text(center[0], center[1], center[2], f'{vertex}', color='k')

    plt.title(title)
    plt.show()

def compare_nerve_and_graph(nerve_complex, G_edges):
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

def run_test_case(G_edges, title="Test Case"):
    G = edge_list_to_adjacency_list(G_edges)
    is_planar = is_planar_graph(G_edges)
    final_embedding_dim = 3 if is_planar else 6  # Use higher dimension for non-planar graphs
    result = assign_positions_and_radii(
        G, initial_embedding_dim=6, final_embedding_dim=final_embedding_dim, num_attempts=5
    )

    if result is not None:
        convex_sets, embedding_dim = result
        nerve_complex = compute_nerve_complex(convex_sets)
        visualize_convex_sets(convex_sets, embedding_dim, title=title)
        test_passed = compare_nerve_and_graph(nerve_complex, G_edges)

        print("Nerve complex simplices:")
        for simplex in sorted(nerve_complex, key=lambda s: (len(s[0]), s[0])):
            print(simplex)
    else:
        print("Failed to construct convex sets for", title)
        test_passed = False

    return test_passed

def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)  # Suppress MDS warnings

    # test cases
    np.random.seed(42)
    positions = np.random.rand(20, 2)
    r = 0.3
    edges_19 = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < r:
                edges_19.append((i, j))

    # random graphs
    np.random.seed(123)
    G_sparse_random = nx.gnp_random_graph(30, 0.05)
    edges_sparse_random = list(G_sparse_random.edges())

    G_dense_random = nx.gnp_random_graph(20, 0.5)
    edges_dense_random = list(G_dense_random.edges())

    G_binary_tree = nx.balanced_tree(r=2, h=4)
    edges_binary_tree = list(G_binary_tree.edges())

    G_cycle_10 = nx.cycle_graph(10)
    edges_cycle_10 = list(G_cycle_10.edges())

    G_line_15 = nx.path_graph(15)
    edges_line_15 = list(G_line_15.edges())

    G_star_10 = nx.star_graph(10)
    edges_star_10 = list(G_star_10.edges())

    G_bipartite = nx.complete_bipartite_graph(3, 5)
    edges_bipartite = list(G_bipartite.edges())

    G_planar_grid = nx.grid_2d_graph(5, 5)
    G_planar_grid = nx.convert_node_labels_to_integers(G_planar_grid)
    edges_planar_grid = list(G_planar_grid.edges())

    G_multiple_components = nx.Graph()
    G_multiple_components.add_edges_from([
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (6, 7), (7, 8), (8, 6)
    ])
    edges_multiple_components = list(G_multiple_components.edges())


    G_groetzsch = nx.mycielski_graph(4)
    G_groetzsch = nx.convert_node_labels_to_integers(G_groetzsch)
    edges_groetzsch = list(G_groetzsch.edges())

    from itertools import combinations
    subsets = list(combinations(range(5), 2))
    num_vertices = len(subsets)
    edges_kneser = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if set(subsets[i]).isdisjoint(subsets[j]):
                edges_kneser.append((i, j))

    edges_fano_plane = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (5, 6),
        (0, 5), (0, 6)
    ]

    test_cases = [
        {
            'title': 'Test Case 1: Simple Triangle Graph',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 0)
            ]
        },
        {
            'title': 'Test Case 2: Star Graph',
            'edges': [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4)
            ]
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
            ]
        },
        {
            'title': 'Test Case 4: Linear Chain',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4)
            ]
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
            ]
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
            ]
        },
        {
            'title': 'Test Case 7: Cycle with a Chord',
            'edges': [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 2)
            ]
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
            ]
        },
        {
            'title': 'Test Case 9: Petersen Graph',
            'edges': list(nx.petersen_graph().edges())
        },
        {
            'title': 'Test Case 10: 3D Hypercube Graph',
            'edges': list(nx.convert_node_labels_to_integers(nx.hypercube_graph(3)).edges())
        },
        {
            'title': 'Test Case 11: Complete Bipartite Graph K3,3',
            'edges': list(nx.complete_bipartite_graph(3, 3).edges())
        },
        {
            'title': 'Test Case 12: Wheel Graph W6',
            'edges': list(nx.wheel_graph(6).edges())
        },
        {
            'title': 'Test Case 13: Ladder Graph',
            'edges': list(nx.ladder_graph(2).edges())
        },
        {
            'title': 'Test Case 14: Grid Graph 3x3',
            'edges': list(nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)).edges())
        },
        {
            'title': 'Test Case 15: Barbell Graph',
            'edges': list(nx.barbell_graph(3, 0).edges())
        },
        {
            'title': 'Test Case 16: Path Graph P5',
            'edges': list(nx.path_graph(5).edges())
        },
        {
            'title': 'Test Case 17: Cycle Graph C6',
            'edges': list(nx.cycle_graph(6).edges())
        },
        {
            'title': 'Test Case 18: Friendship Graph with 5 triangles',
            'edges': list(nx.convert_node_labels_to_integers(nx.windmill_graph(5, 3)).edges())
        },
        {
            'title': 'Test Case 19: Random Geometric Graph G(20, r=0.3)',
            'edges': edges_19
        },
        {
            'title': 'Test Case 20: 4D Hypercube Graph',
            'edges': list(nx.convert_node_labels_to_integers(nx.hypercube_graph(4)).edges())
        },
        {
            'title': 'Test Case 21: Large Sparse Random Graph',
            'edges': edges_sparse_random
        },
        {
            'title': 'Test Case 22: Dense Random Graph',
            'edges': edges_dense_random
        },
        {
            'title': 'Test Case 23: Binary Tree of Depth 4',
            'edges': edges_binary_tree
        },
        {
            'title': 'Test Case 24: Complete Graph K5',
            'edges': list(nx.complete_graph(5).edges())
        },
        {
            'title': 'Test Case 25: Cycle Graph with 10 Nodes',
            'edges': edges_cycle_10
        },
        {
            'title': 'Test Case 26: Line Graph with 15 Nodes',
            'edges': edges_line_15
        },
        {
            'title': 'Test Case 27: Star Graph with 10 Leaves',
            'edges': edges_star_10
        },
        {
            'title': 'Test Case 28: Disconnected Graph with Multiple Components',
            'edges': edges_multiple_components
        },
        {
            'title': 'Test Case 29: Bipartite Graph K3,5',
            'edges': edges_bipartite
        },
        {
            'title': 'Test Case 30: Planar Grid Graph 5x5',
            'edges': edges_planar_grid
        },
        {
            'title': 'Test Case 31: Grötzsch Graph',
            'edges': edges_groetzsch
        },
        {
            'title': 'Test Case 32: Kneser Graph K₅,₂',
            'edges': edges_kneser
        },
        {
            'title': 'Test Case 33: Graph with Impossible Constraints',
            'edges': edges_fano_plane
        }
    ]

    total_tests = len(test_cases)
    passed_tests = 0
    failed_tests = 0
    failed_test_cases = []

    for test_case in test_cases:
        print("\n" + "="*50)
        print(test_case['title'])
        print("="*50)
        test_passed = run_test_case(test_case['edges'], title=test_case['title'])
        if test_passed:
            passed_tests += 1
        else:
            failed_tests += 1
            failed_test_cases.append(test_case['title'])

    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {failed_tests}")

    if failed_tests > 0:
        print("Failed Test Cases:")
        for title in failed_test_cases:
            print(f"- {title}")

if __name__ == "__main__":
    main()
