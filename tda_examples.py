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
        },
        {
            'title': 'Test Case 9: Petersen Graph',
            'edges': [
                (0, 1), (0, 4), (0, 5),
                (1, 2), (1, 6),
                (2, 3), (2, 7),
                (3, 4), (3, 8),
                (4, 9),
                (5, 7), (5, 8),
                (6, 8), (6, 9),
                (7, 9)
            ]
        },
        {
            'title': 'Test Case 10: 3D Hypercube Graph',
            'edges': [
                (0, 1), (0, 3), (0, 4),
                (1, 2), (1, 5),
                (2, 3), (2, 6),
                (3, 7),
                (4, 5), (4, 7),
                (5, 6),
                (6, 7)
            ]
        },
        {
            'title': 'Test Case 11: Complete Bipartite Graph K3,3',
            'edges': [
                (0, 3), (0, 4), (0, 5),
                (1, 3), (1, 4), (1, 5),
                (2, 3), (2, 4), (2, 5)
            ]
        },

        # Test Case 12: Wheel Graph W6
        {
            'title': 'Test Case 12: Wheel Graph W6',
            'edges': (

                [(i, (i + 1) % 5) for i in range(5)] +

                [(5, i) for i in range(5)]
            )
        },
        # Test Case 13: Grid Graph 3x3
        {
            'title': 'Test Case 13: Grid Graph 3x3',
            'edges': [
                # Horizontal edges
                (0, 1), (1, 2),
                (3, 4), (4, 5),
                (6, 7), (7, 8),
                # Vertical edges
                (0, 3), (3, 6),
                (1, 4), (4, 7),
                (2, 5), (5, 8)
            ]
        },
        # Test Case 14: Erdős–Rényi Random Dense Graph G(10, 0.7)
        {
            'title': 'Test Case 14: Erdős–Rényi Random Dense Graph G(10, 0.7)',
            'edges': [
                (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),
                (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),
                (2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),
                (3,4),(3,5),(3,6),(3,7),(3,8),(3,9),
                (4,5),(4,6),(4,7),(4,8),(4,9),
                (5,6),(5,7),(5,8),(5,9),
                (6,7),(6,8),(6,9),
                (7,8),(7,9),
                (8,9)
            ]
        },
        # Test Case 15: Barabási–Albert Scale-Free Graph (15 nodes, m=2)
        {
            'title': 'Test Case 15: Barabási–Albert Scale-Free Graph (15 nodes, m=2)',
            'edges': (
                # Start with a triangle
                [(0,1),(1,2),(2,0)] +
                # Attach new nodes with m=2 edges each
                [(3,1),(3,2),
                 (4,1),(4,3),
                 (5,2),(5,3),
                 (6,2),(6,5),
                 (7,5),(7,6),
                 (8,5),(8,7),
                 (9,7),(9,8),
                 (10,7),(10,9),
                 (11,9),(11,10),
                 (12,10),(12,11),
                 (13,11),(13,12),
                 (14,12),(14,13)]
            )
        },
        # Test Case 16: Lollipop Graph Lollipop(5,4)
        {
            'title': 'Test Case 16: Lollipop Graph Lollipop(5,4)',
            'edges': (
                # Complete graph K5
                [(0,1),(0,2),(0,3),(0,4),
                 (1,2),(1,3),(1,4),
                 (2,3),(2,4),
                 (3,4)] +

                [(4,5),(5,6),(6,7),(7,8)]
            )
        },
        # Test Case 17: Platonic Solid - Icosahedron
        {
            'title': 'Test Case 17: Platonic Solid - Icosahedron',
            'edges': [
                (0,1),(0,4),(0,5),
                (1,0),(1,2),(1,6),
                (2,1),(2,3),(2,7),
                (3,2),(3,4),(3,8),
                (4,0),(4,3),(4,9),
                (5,0),(5,7),(5,8),
                (6,1),(6,9),(6,10),
                (7,2),(7,5),(7,11),
                (8,3),(8,5),(8,11),
                (9,4),(9,6),(9,10),
                (10,6),(10,9),(10,11),
                (11,7),(11,8),(11,10)
            ]
        },
        # Test Case 18: Friendship Graph (Windmill Graph) with 5 triangles
        {
            'title': 'Test Case 18: Friendship Graph (Windmill Graph) with 5 triangles',
            'edges': (

                [(0,1),(0,2),
                 (0,3),(0,4),
                 (0,5),(0,6),
                 (0,7),(0,8),
                 (0,9),(0,10)]
            )
        },
        # Test Case 19: Random Geometric Graph G(20, r=0.3)
        {
            'title': 'Test Case 19: Random Geometric Graph G(20, r=0.3)',
            'edges': [

                (0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),
                (5,7),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),
                (12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
                (18,19),(19,0)
            ]
        },
        # Test Case 20: 4D Hypercube Graph
        {
            'title': 'Test Case 20: 4D Hypercube Graph',
            'edges': [
                (i, j) for i in range(16) for j in range(i+1,16) if bin(i ^ j).count('1') == 1
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
