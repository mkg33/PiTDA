import numpy as np
from gudhi import SimplexTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def edge_list_to_adjacency_list(edge_list):
    adjacency = {}
    for u, v in edge_list:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    return adjacency

# Example as an edge list
G_edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
]

G = edge_list_to_adjacency_list(G_edges)

def assign_positions_in_high_dimensional_space(G):
    n = len(G)
    positions = {}
    for idx, vertex in enumerate(G.keys()):
        position = np.zeros(n)
        position[idx] = 1
        positions[vertex] = position
    return positions

def define_convex_sets_high_dim(positions):
    convex_sets = {}
    radius = 1.0  # start radius
    for vertex, position in positions.items():
        convex_sets[vertex] = {
            'center': position,
            'radius': radius
        }
    return convex_sets

def adjust_radii(convex_sets, G):
    for u in G:
        for v in G:
            if u >= v:
                continue
            set_u = convex_sets[u]
            set_v = convex_sets[v]
            distance = np.linalg.norm(set_u['center'] - set_v['center'])
            if v in G[u]:
                # Adjacent vertices should overlap
                required_radius = distance / 2 + 0.1
            else:
                # But non-adjacent vertices should not
                required_radius = distance / 2 - 0.1
            set_u['radius'] = max(set_u['radius'], required_radius)
            set_v['radius'] = max(set_v['radius'], required_radius)

def project_convex_sets_to_r3(convex_sets_high_dim):
    centers = np.array([convex_set['center'] for convex_set in convex_sets_high_dim.values()])
    pca = PCA(n_components=3)
    centers_r3 = pca.fit_transform(centers)

    convex_sets_r3 = {}
    for idx, (vertex, convex_set) in enumerate(convex_sets_high_dim.items()):
        center_r3 = centers_r3[idx]
        convex_sets_r3[vertex] = {
            'center': center_r3,
            'radius': convex_set['radius']
        }
    return convex_sets_r3

def adjust_radii_after_projection(convex_sets, G):
    for u in G:
        for v in G:
            if u >= v:
                continue
            set_u = convex_sets[u]
            set_v = convex_sets[v]
            center_u = set_u['center']
            center_v = set_v['center']
            distance = np.linalg.norm(center_u - center_v)
            if v in G[u]:
                # Adjacent vertices should overlap
                required_radius = distance / 2 + 0.1
            else:
                # But non-adjacent vertices should not
                required_radius = distance / 2 - 0.1
            set_u['radius'] = max(set_u['radius'], required_radius)
            set_v['radius'] = max(set_v['radius'], required_radius)

def check_overlap(set_a, set_b):
    center_a = set_a['center']
    center_b = set_b['center']
    radius_a = set_a['radius']
    radius_b = set_b['radius']
    distance = np.linalg.norm(center_a - center_b)
    return distance < (radius_a + radius_b)

def compute_nerve_complex(convex_sets):
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    # start with 0-simplices
    for v in vertices:
        simplex_tree.insert([v])

    # If convex sets overlap, add 1-simplices
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            v_i = vertices[i]
            v_j = vertices[j]
            if check_overlap(convex_sets[v_i], convex_sets[v_j]):
                simplex_tree.insert([v_i, v_j])

    return simplex_tree

def visualize_convex_sets(convex_sets):
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

    plt.title('Convex Sets in R3')
    plt.show()

positions_high_dim = assign_positions_in_high_dimensional_space(G)
convex_sets_high_dim = define_convex_sets_high_dim(positions_high_dim)
adjust_radii(convex_sets_high_dim, G)
convex_sets_r3 = project_convex_sets_to_r3(convex_sets_high_dim)
adjust_radii_after_projection(convex_sets_r3, G)
nerve_complex = compute_nerve_complex(convex_sets_r3)

visualize_convex_sets(convex_sets_r3)

print("Nerve complex simplices:")
for simplex in nerve_complex.get_simplices():
    print(simplex)
