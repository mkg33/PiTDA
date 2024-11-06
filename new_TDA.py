import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gudhi import SimplexTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# potential problem: everything depends on the initial optimization result

# assumption: convex polytope
# check weird examples
# convex polytopes in R4 (nerve) -> projection
# pick random vertices of convex polytope

def edge_list_to_adjacency_list(edge_list):
    adjacency = {}
    for u, v in edge_list:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    return adjacency

# placeholder for input
G_edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
]

G = edge_list_to_adjacency_list(G_edges)

# embed in R4 first using optimization
def assign_positions_and_radii(G, embedding_dim=4):
    vertices = list(G.keys())
    n = len(vertices)

    projection_dim = 3

    # guess initial positions
    positions = np.random.rand(n, embedding_dim)

    # set initial radii
    radii = np.ones(n) * 0.5

    x0 = np.hstack([positions.flatten(), radii])

    adjacency = G  # the placeholder input is already an AL, but this step is needed for processing custom inputs

    def objective(x):
        # goal: minimize the sum of radii (prevents large spheres that aren't needed)
        radii = x[n * embedding_dim:]
        return np.sum(radii)

    def constraints(x):
        positions = x[:n * embedding_dim].reshape((n, embedding_dim))
        radii = x[n * embedding_dim:]
        cons = []
        epsilon = 1e-6

        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                dist = np.linalg.norm(positions[i] - positions[j])
                r_sum = radii[i] + radii[j]

                if v in adjacency[u]:
                    # overlap
                    cons.append(r_sum - dist - epsilon)
                else:
                    # no overlap
                    cons.append(dist - r_sum - epsilon)
        return np.array(cons)

    bounds = [(-np.inf, np.inf)] * (n * embedding_dim) + [(0.1, None)] * n  # radii must be >= 0.1 (to prevent collapsing to a point)

    cons = {'type': 'ineq', 'fun': constraints}

    # we use sequential least squares programming from scipy
    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 5000, 'ftol': 1e-9}
    )

    if not res.success:
        print("Optimization failed :-(... :", res.message)
        return None

    x_opt = res.x
    positions_opt = x_opt[:n * embedding_dim].reshape((n, embedding_dim))
    radii_opt = x_opt[n * embedding_dim:]

    # project to R3 via PCA
    if embedding_dim > projection_dim:
        pca = PCA(n_components=projection_dim)
        positions_proj = pca.fit_transform(positions_opt)
    else:
        positions_proj = positions_opt

    convex_sets = {
        vertices[i]: {'center': positions_proj[i], 'radius': radii_opt[i]}
        for i in range(n)
    }

    return convex_sets

# TODO: combine the functions!
def refine_positions_and_radii_in_R3(convex_sets, G):
    vertices = list(convex_sets.keys())
    n = len(vertices)

    positions = np.array([convex_sets[v]['center'] for v in vertices])
    radii = np.array([convex_sets[v]['radius'] for v in vertices])

    embedding_dim = 3

    x0 = np.hstack([positions.flatten(), radii])

    adjacency = G # already AL

    def objective(x): # copied from the original function
        radii = x[n * embedding_dim:]
        return np.sum(radii)

    def constraints(x): # same
        positions = x[:n * embedding_dim].reshape((n, embedding_dim))
        radii = x[n * embedding_dim:]
        cons = []
        epsilon = 1e-6

        for i in range(n):
            for j in range(i + 1, n):
                u = vertices[i]
                v = vertices[j]
                dist = np.linalg.norm(positions[i] - positions[j])
                r_sum = radii[i] + radii[j]

                if v in adjacency[u]:
                    # overlap
                    cons.append(r_sum - dist - epsilon)
                else:
                    # no overlap
                    cons.append(dist - r_sum - epsilon)
        return np.array(cons)

    bounds = [(-np.inf, np.inf)] * (n * embedding_dim) + [(0.1, None)] * n  # Radii >= 0.1

    cons = {'type': 'ineq', 'fun': constraints}

    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 5000, 'ftol': 1e-9}
    )

    if not res.success:
        print("Failed! :", res.message)
        return None

    x_opt = res.x
    positions_opt = x_opt[:n * embedding_dim].reshape((n, embedding_dim))
    radii_opt = x_opt[n * embedding_dim:]

    convex_sets_refined = {
        vertices[i]: {'center': positions_opt[i], 'radius': radii_opt[i]}
        for i in range(n)
    }

    return convex_sets_refined

def compute_nerve_complex(convex_sets):
    simplex_tree = SimplexTree()
    vertices = list(convex_sets.keys())

    # 0-simplices
    for v in vertices:
        simplex_tree.insert([v])

    # If overlap, insert 1-simplices
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

    plt.title('R3: convex sets')
    plt.show()

convex_sets_R4 = assign_positions_and_radii(G, embedding_dim=4)
if convex_sets_R4 is not None:

    convex_sets_R3 = refine_positions_and_radii_in_R3(convex_sets_R4, G)

    if convex_sets_R3 is not None:
        nerve_complex = compute_nerve_complex(convex_sets_R3)

        visualize_convex_sets(convex_sets_R3)

        print("Nerve complex simplices: ")
        for simplex in nerve_complex.get_simplices():
            print(simplex)
    else:
        print("Failed to project to R3...")
else:
    print("Failed to construct convex sets in R4...")
