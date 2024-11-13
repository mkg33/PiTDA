import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gudhi import SimplexTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS

def edge_list_to_adjacency_list(edge_list):
    adjacency = {}
    for u, v in edge_list:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    return adjacency

def assign_positions_and_radii(G, embedding_dim=4, num_attempts=10, penalty_weight=1000):
    vertices = list(G.keys())
    n = len(vertices)

    best_positions = None
    best_radii = None
    best_obj = np.inf

    adjacency = G  #

    for attempt in range(num_attempts):
        # assign random initial positions and radii
        positions = np.random.rand(n, embedding_dim)
        radii = np.random.rand(n) * 0.5 + 0.1  # radii: from 0.1 to 0.6

        x0 = np.hstack([positions.flatten(), radii])

        def objective(x):
            positions = x[:n * embedding_dim].reshape((n, embedding_dim))
            radii = x[n * embedding_dim:]
            obj = np.sum(radii)
            penalty = 0
            for i in range(n):
                for j in range(i + 1, n):
                    u = vertices[i]
                    v = vertices[j]
                    dist = np.linalg.norm(positions[i] - positions[j])
                    r_sum = radii[i] + radii[j]
                    if v in adjacency[u]:
                        # overlap
                        penalty += penalty_weight * max(0, dist - r_sum + 1e-6) ** 2
                    else:
                        # no overlap
                        penalty += penalty_weight * max(0, r_sum - dist + 1e-6) ** 2
            return obj + penalty

        bounds = [(-np.inf, np.inf)] * (n * embedding_dim) + [(0.1, 1.0)] * n  # radii

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if res.success:
            if res.fun < best_obj:
                best_obj = res.fun
                x_opt = res.x
                positions_opt = x_opt[:n * embedding_dim].reshape((n, embedding_dim))
                radii_opt = x_opt[n * embedding_dim:]
                best_positions = positions_opt
                best_radii = radii_opt

    if best_positions is None:
        print("Optimization failure after multiple attempts...")
        return None

    # Project to R3 via MDS if embedding_dim > 3
    if embedding_dim > 3:
        mds = MDS(n_components=3, dissimilarity='euclidean', random_state=42)
        positions_proj = mds.fit_transform(best_positions)
    else:
        positions_proj = best_positions

    # assign positions and radii in R3
    embedding_dim = 3
    positions = positions_proj
    radii = best_radii
    x0 = np.hstack([positions.flatten(), radii])

    def objective_refine(x):
        positions = x[:n * embedding_dim].reshape((n, embedding_dim))
        radii = x[n * embedding_dim:]
        obj = np.sum(radii)
        penalty = 0
        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                dist = np.linalg.norm(positions[i] - positions[j])
                r_sum = radii[i] + radii[j]
                if v in adjacency[u]:
                    penalty += penalty_weight * max(0, dist - r_sum + 1e-6) ** 2
                else:
                    penalty += penalty_weight * max(0, r_sum - dist + 1e-6) ** 2
        return obj + penalty

    bounds = [(-np.inf, np.inf)] * (n * embedding_dim) + [(0.1, 1.0)] * n

    res_refine = minimize(objective_refine, x0, method='L-BFGS-B', bounds=bounds)

    if res_refine.success:
        x_opt = res_refine.x
        positions_opt = x_opt[:n * embedding_dim].reshape((n, embedding_dim))
        radii_opt = x_opt[n * embedding_dim:]
    else:
        print("Problem: R3 failed...")
        return None

    convex_sets = {
        vertices[i]: {'center': positions_opt[i], 'radius': radii_opt[i]}
        for i in range(n)
    }

    return convex_sets

def compute_nerve_complex(convex_sets):
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    # 0-simplices
    for v in vertices:
        simplex_tree.insert([v])

    # insert 1-simplices if overlap found
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            v_i = vertices[i]
            v_j = vertices[j]
            set_i = convex_sets[v_i]
            set_j = convex_sets[v_j]
            distance = np.linalg.norm(set_i['center'] - set_j['center'])
            r_sum = set_i['radius'] + set_j['radius']
            if distance < r_sum - 1e-6:
                simplex_tree.insert([v_i, v_j])

    return simplex_tree

def visualize_convex_sets(convex_sets, title="Convex Sets in R3"):
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
    nerve_edges = set()
    for simplex in nerve_complex.get_simplices():
        if len(simplex[0]) == 2:
            nerve_edges.add(tuple(sorted(simplex[0])))
    graph_edges = set(tuple(sorted(edge)) for edge in G_edges)
    missing_edges = graph_edges - nerve_edges
    extra_edges = nerve_edges - graph_edges
    if missing_edges or extra_edges:
        print("Discrepancies found:")
        if missing_edges:
            print("Edges missing in nerve:", missing_edges)
        if extra_edges:
            print("Extra edges in nerve:", extra_edges)
    else:
        print("Nerve complex matches the input graph :-)")

def run_test_case(G_edges, title="Test Case"):
    G = edge_list_to_adjacency_list(G_edges)
    convex_sets = assign_positions_and_radii(
        G, embedding_dim=4, num_attempts=10, penalty_weight=1000
    )

    if convex_sets is not None:
        nerve_complex = compute_nerve_complex(convex_sets)
        visualize_convex_sets(convex_sets, title=title)
        compare_nerve_and_graph(nerve_complex, G_edges)

        print("Nerve complex simplices:")
        for simplex in sorted(nerve_complex.get_simplices()):
            print(simplex)
    else:
        print("Failed to construct convex sets for", title)

def main():
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
                # First triangle
                (0, 1),
                (1, 2),
                (2, 0),
                # Second triangle
                (3, 4),
                (4, 5),
                (5, 3)
            ]
        }
    ]

    for test_case in test_cases:
        print("\n" + "="*50)
        print(test_case['title'])
        print("="*50)
        run_test_case(test_case['edges'], title=test_case['title'])

if __name__ == "__main__":
    main()
