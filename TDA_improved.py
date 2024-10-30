import numpy as np
from gudhi import SimplexTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

def edge_list_to_adjacency_list(edge_list):
    adjacency = {}
    for u, v in edge_list:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    return adjacency

# this is just an example
G_edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
]

G = edge_list_to_adjacency_list(G_edges)

def assign_positions_in_R4(G):
    vertices = list(G.keys())
    n = len(vertices)

    # fill the dissimilarity matrix
    dissimilarity_matrix = np.zeros((n, n))
    for i, u in enumerate(vertices):
        for j, v in enumerate(vertices):
            if u == v:
                dissimilarity_matrix[i, j] = 0
            elif v in G[u]:
                dissimilarity_matrix[i, j] = 1  # adjacent nodes: small dissimilarity
            else:
                dissimilarity_matrix[i, j] = 2  # non-adjacent nodes: larger dissimilarity

    # embed the graph into R^4
    mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
    positions_array = mds.fit_transform(dissimilarity_matrix)

    positions = {vertex: positions_array[i] for i, vertex in enumerate(vertices)}
    return positions

def define_convex_sets_R4(positions):
    convex_sets = {}
    for vertex, position in positions.items():
        convex_sets[vertex] = {
            'center': position,
            'radius': 0.1  # experiment with different values (TODO)
        }
    return convex_sets

def adjust_radii_R4(convex_sets, G):
    vertices = list(convex_sets.keys())
    for i, u in enumerate(vertices):
        for j, v in enumerate(vertices):
            if i >= j:
                continue
            set_u = convex_sets[u]
            set_v = convex_sets[v]
            distance = np.linalg.norm(set_u['center'] - set_v['center'])
            if v in G[u]:
                required_radius = distance / 2 + 0.1  # overlap
            else:
                required_radius = distance / 2 - 0.1  # non-overlap
                if required_radius < 0:
                    required_radius = 0.1
            set_u['radius'] = max(set_u['radius'], required_radius)
            set_v['radius'] = max(set_v['radius'], required_radius)

# use PCA for projection
def project_convex_sets_to_R3(convex_sets_R4):
    centers = np.array([convex_set['center'] for convex_set in convex_sets_R4.values()])
    pca = PCA(n_components=3)
    centers_R3 = pca.fit_transform(centers)

    convex_sets_R3 = {}
    for idx, (vertex, convex_set) in enumerate(convex_sets_R4.items()):
        center_R3 = centers_R3[idx]
        convex_sets_R3[vertex] = {
            'center': center_R3,
            'radius': convex_set['radius']
        }
    return convex_sets_R3

def adjust_radii_after_projection(convex_sets, G):
    vertices = list(convex_sets.keys())
    for i, u in enumerate(vertices):
        for j, v in enumerate(vertices):
            if i >= j:
                continue
            set_u = convex_sets[u]
            set_v = convex_sets[v]
            center_u = set_u['center']
            center_v = set_v['center']
            distance = np.linalg.norm(center_u - center_v)
            if v in G[u]:
                required_radius = distance / 2 + 0.1
            else:
                required_radius = distance / 2 - 0.1
                if required_radius < 0:
                    required_radius = 0.1
            set_u['radius'] = max(set_u['radius'], required_radius)
            set_v['radius'] = max(set_v['radius'], required_radius)

def compute_nerve_complex(convex_sets):
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    # 0-simplices
    for v in vertices:
        simplex_tree.insert([v])

    # If convex sets overlap, insert 1-simplices
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            v_i = vertices[i]
            v_j = vertices[j]
            set_i = convex_sets[v_i]
            set_j = convex_sets[v_j]
            distance = np.linalg.norm(set_i['center'] - set_j['center'])
            if distance < (set_i['radius'] + set_j['radius']):
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

positions_R4 = assign_positions_in_R4(G)

convex_sets_R4 = define_convex_sets_R4(positions_R4)

adjust_radii_R4(convex_sets_R4, G)

convex_sets_R3 = project_convex_sets_to_R3(convex_sets_R4)

adjust_radii_after_projection(convex_sets_R3, G)

nerve_complex = compute_nerve_complex(convex_sets_R3)

visualize_convex_sets(convex_sets_R3)

print("Nerve complex simplices:")
for simplex in nerve_complex.get_simplices():
    print(simplex)
