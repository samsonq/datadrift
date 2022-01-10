from queue import Queue
import numpy as np
from copy import deepcopy
import pdb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class KDQNode:

    def __init__(self, data, limits: list, dim_split: int, delta: float, min_samples: int, node_id: int):
        self.data = data
        self.size = len(data)
        self.limits = limits
        self.dim_split = dim_split
        self.delta = delta
        self.min_samples = min_samples
        self.id = node_id
        self.left = None
        self.right = None


class KDQTree:
    def __init__(self, points, min_samples, delta):

        """try:
            assert points.max() <=1 and points.min() >= 0
        except ValueError:
            print("Your data is not in [0;1]. Please rescale before initiating.")"""

        self.dim = points.shape[1]
        self.points = points
        self.min_samples = min_samples
        self.delta = delta
        self.num_nodes = 1
        self.num_leaves = 1
        self.root = KDQNode(np.arange(len(points)),
                            [(0, 1) for i in range(self.dim)],
                            0, self.delta, self.min_samples, 0)

    def split_node(self, node: KDQNode):
        # print("Starting split")
        max_width = max([b - a for (a, b) in node.limits])
        size = node.size
        if size < self.min_samples or max_width < self.delta:
            return []
        else:
            dim = node.dim_split
            limits = node.limits[dim]
            cutoff = (limits[1] + limits[0]) / 2
            # pdb.set_trace()
            while limits[1] - limits[0] < self.delta:
                dim = (dim + 1) % self.dim
                limits = node.limits[dim]
                cutoff = (limits[1] + limits[0]) / 2
            # print(cutoff)
            node.dim = dim
            data_loc = node.data

            # Split down the middle
            left_points = data_loc[self.points[data_loc, dim] <= cutoff]
            right_points = data_loc[self.points[data_loc, dim] > cutoff]
            if len(left_points) == 0 or len(right_points) == 0:
                return []

            # Create left node
            self.num_leaves += 1
            left_limits = deepcopy(node.limits)
            left_limits[dim] = (limits[0], cutoff)
            left_node = KDQNode(left_points, left_limits, (dim + 1) % self.dim,
                                self.delta, self.min_samples, self.num_nodes)
            self.num_nodes += 1
            # Create right node
            right_limits = deepcopy(node.limits)
            right_limits[dim] = (cutoff, limits[1])
            right_node = KDQNode(right_points, right_limits, (dim + 1) % self.dim,
                                 self.delta, self.min_samples, self.num_nodes)
            self.num_nodes += 1

            # Build the links
            node.left = left_node
            node.right = right_node
            return [left_node, right_node]

    def build_tree(self):
        node_queue = Queue()
        node_queue.put(self.root)
        node_counts = {}
        node_counts[self.root.id] = self.root.size

        while not node_queue.empty():
            node = node_queue.get()
            split = self.split_node(node)
            # print(node.id)
            # print(node.limits)
            if len(split) == 2:
                del node_counts[node.id]
                node_counts[split[0].id] = split[0].size
                node_queue.put(split[0])
                node_counts[split[1].id] = split[1].size
                node_queue.put(split[1])

        self.node_counts = node_counts

    def find_leaf(self, point):
        node = self.root
        while node.left is not None or node.right is not None:
            dim = node.dim_split
            cutoff = (node.limits[dim][1] + node.limits[dim][0]) / 2
            if point[dim] >= cutoff:
                node = node.right
            else:
                node = node.left
        return node

    def populate_tree(self, points):
        leaf_counts = {node_id: 0 for node_id in self.node_counts.keys()}
        for point in points:
            node = self.find_leaf(point)
            leaf_counts[node.id] += 1
        return leaf_counts

    def create_probs(self):
        """
        Create a probability distribution based on the leaf counts and
        Laplace smoothing to remove 0 probabilities.

        A dictionary leaf_probs = {leaf: probability} will be set as a parameter of the class.
        """
        A = self.num_leaves
        n = len(self.points)
        # print(A)
        probs = {node_id: (count + 0.5) / (n + A / 2) for (node_id, count) in self.node_counts.items()}
        self.leaf_probs = probs

    def compute_probs(self, points):
        """
        Populate the tree leaves with a new set of points.
        Create a probability distribution based on the leaf counts and
        Laplace smoothing to remove 0 probabilities.

        Return the dictionary of {leaf: probability}
        """
        leaf_counts = self.populate_tree(points)
        A = self.num_leaves
        n = len(self.points)
        return {node_id: (count + 0.5) / (n + A / 2) for (node_id, count) in leaf_counts.items()}

    def compute_divergence(self, points):
        """
        Compute divergence between points tree was built on
        and a new set of points, as in the shortcut in Section 5, page 10.
        """
        base_p = self.node_counts
        new_p = self.populate_tree(points)
        L = self.num_leaves
        n = len(self.points)
        m = len(points)
        div = np.log((m + L / 2) / (n + L / 2))
        # print(div)
        div += np.sum([(base_p[node_id] + 0.5) * np.log((base_p[node_id] + 0.5) / (new_p[node_id] + 0.5)) for node_id in
                       base_p.keys()]) / (n + L / 2)
        return div


def sliding_const_window(data, dates, **kwargs):
    """
    Sliding window algorithm with constant base data. Useful for detecting gradual changes in the data.
    Data is rescaled to [0;1] and then at each step the window of samples is used to construct
    a partition of the cube [0;1]^d. A distribution is constructed and any new datasets are compared to
    that distribution. Implementation of www.cse.ust.hk/~yike/datadiff/datadiff.pdf

    Parameters:
        data: expected numerical values in matrix format.
        kwargs: parameters for the algorithm and the tree data structure
            n: Half the size of the moving window.
            shift: Number of points skipped over each time we slide the window
            n_bootstraps: Number of bootstraps to perform at each step to
                determine the (1-alpha) quantile of the distribution of distances
            alpha: distance d is compared to the (1-alpha)-quantile of the set of
                bootstrapped distances.
            gamma:The persistence parameter. Need gamma*n consecutive success to declare a change.
            delta: The minimum dimension of a cell in the underlying data structure.
            min_samples: Minimal number of samples to have in each cell of the data structure.

    """
    n = kwargs['n']
    shift = kwargs['shift']
    n_bootstraps = kwargs["n_bootstraps"]
    alpha = kwargs["alpha"]
    gamma = kwargs['gamma']
    delta = kwargs['delta']
    min_samples = kwargs['min_samples']
    N = len(data)
    t = 2 * n

    # Rescale data
    scaler = MinMaxScaler()
    scaler.fit(data)
    drift_data = scaler.transform(data)

    KLs = np.zeros((N - 2 * n) // shift + 1)

    Base = drift_data[:n]
    New = drift_data[n:2 * n]
    tree = KDQTree(Base, min_samples, delta)
    tree.build_tree()
    tree.create_probs()
    KLs[0] = tree.compute_divergence(New)
    count = 0
    changes = []
    for t in tqdm(range(2 * n, N, shift)):
        New = drift_data[t - n:t]

        # print(tree.num_leaves)
        # print((t-2*n)//shift)
        d = tree.compute_divergence(New)
        KLs[(t - 2 * n) // shift] = d

        # Perform bootstrap
        probs = tree.leaf_probs
        prob = list(probs.values())
        leaves = list(probs.keys())
        Sample_D = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            L = len(leaves)
            sample = np.random.choice(a=leaves, size=2 * n, p=prob)
            Base = sample[:n]
            base_counts = np.unique(Base, return_counts=True)
            base_counts = dict(zip(base_counts[0], base_counts[1]))
            base_p = {node_id: (base_counts.get(node_id, 0) + 0.5) / (n + L / 2) for node_id in leaves}
            New = sample[n:]
            new_counts = np.unique(New, return_counts=True)
            new_counts = dict(zip(new_counts[0], new_counts[1]))
            new_p = {node_id: (new_counts.get(node_id, 0) + 0.5) / (n + L / 2) for node_id in leaves}
            # print(new_counts.keys())
            Sample_D[i] = np.sum([base_p[node_id] * np.log(base_p[node_id] / new_p[node_id]) for node_id in leaves])
        d_q = np.quantile(Sample_D, q=alpha)
        if d >= d_q:
            count += 1
        else:
            count = 0
        if count >= gamma * n:
            changes.append(t)
            count = 0
            Base = drift_data[t - 2 * n:t - n]
            tree = KDQTree(Base, min_samples, delta)
            tree.build_tree()
            tree.create_probs()

            print(f"Change detected at location {t}")
    plt.figure(figsize=(20, 8))
    plt.title("KL value for different locations in the data")
    locations = list(range(2 * n, N, shift))
    date_ticks = pd.Series({change: str(dates.iloc[change])[:10] for change in changes})
    plt.xticks(date_ticks.index, date_ticks.values, size='small')
    plt.ylabel("Value of distance metric")
    for change in changes:
        plt.axvline(change, color="red", label=f"change detected here at {change}")

    plt.plot(list(range(2 * n, N, shift)), KLs)
    plt.legend()
    return {"KL_values": KLs, "changes": changes}


def sliding_adjoint_window(data, dates, **kwargs):
    """
    Sliding window algorithm. Useful for detecting local sudden changes in the data.
    Data is rescaled to [0;1] and then at each step the window of samples is used to construct
    a partition of the cube [0;1]^d. A distribution is constructed and any new datasets are compared to
    that distribution. Plots the series of distances as well as any locations where change is detected.

    Implementation of www.cse.ust.hk/~yike/datadiff/datadiff.pdf

    Parameters:
        data: expected numerical values in matrix format.
        kwargs: parameters for the algorithm and the tree data structure
            n: Half the size of the moving window.
            shift: Number of points skipped over each time we slide the window
            n_bootstraps: Number of bootstraps to perform at each step to
                determine the (1-alpha) quantile of the distribution of distances
            alpha: distance d is compared to the (1-alpha)-quantile of the set of
                bootstrapped distances.
            gamma:The persistence parameter. Need gamma*n consecutive success to declare a change.
            delta: The minimum dimension of a cell in the underlying data structure.
            min_samples: Minimal number of samples to have in each cell of the data structure.
    Returns:
        The distances at every point in the data where we performed the test.
        Locations of change in the data(array indices, not index locations in a dataframe)


    """
    n = kwargs['n']
    shift = kwargs['shift']
    n_bootstraps = kwargs["n_bootstraps"]
    alpha = kwargs["alpha"]
    gamma = kwargs['gamma']
    delta = kwargs['delta']
    min_samples = kwargs['min_samples']
    scaler = MinMaxScaler()
    scaler.fit(data)
    drift_data = scaler.transform(data)

    N = len(drift_data)
    KLs = np.zeros((N - 2 * n) // shift + 1)
    t = 2 * n
    Base = drift_data[t - 2 * n:t - n]
    New = drift_data[t - n:t]
    tree = KDQTree(Base, min_samples, delta)
    tree.build_tree()
    KLs[0] = tree.compute_divergence(New)
    count = 0
    changes = []
    for t in tqdm(range(2 * n, N, shift)):

        Base = drift_data[t - 2 * n:t - n]
        New = drift_data[t - n:t]

        # Build new KDQ Tree
        tree = KDQTree(Base, min_samples, delta)
        tree.build_tree()
        tree.create_probs()
        d = tree.compute_divergence(New)
        KLs[(t - 2 * n) // shift] = d

        # Perform bootstrap
        probs = tree.leaf_probs
        prob = list(probs.values())
        leaves = list(probs.keys())
        Sample_D = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            L = len(leaves)
            # Bootstrap from distribution over leaves
            sample = np.random.choice(a=leaves, size=2 * n, p=prob)
            # Compare the first n and last n samples
            Base = sample[:n]
            base_counts = np.unique(Base, return_counts=True)
            base_counts = dict(zip(base_counts[0], base_counts[1]))
            base_p = {node_id: (base_counts.get(node_id, 0) + 0.5) / (n + L / 2) for node_id in leaves}

            New = sample[n:]
            new_counts = np.unique(New, return_counts=True)
            new_counts = dict(zip(new_counts[0], new_counts[1]))
            new_p = {node_id: (new_counts.get(node_id, 0) + 0.5) / (n + L / 2) for node_id in leaves}
            # KL distance between New and Base
            Sample_D[i] = np.sum([base_p[node_id] * np.log(base_p[node_id] / new_p[node_id]) for node_id in leaves])
        d_q = np.quantile(Sample_D, q=alpha)
        # Modify count and declare a change depending on the value of d and d_q
        if d >= d_q:
            count += 1
        else:
            count = 0
        if count >= gamma * n:
            changes.append(t - n)
            count = 0

            print(f"Change detected at location {t}")

    plt.figure(figsize=(20, 8))
    plt.title("KL value for different locations in the data")
    locations = list(range(2 * n, N, shift))
    date_ticks = pd.Series({change: str(dates.iloc[change])[:10] for change in changes})
    plt.xticks(date_ticks.index, date_ticks.values, size='small')
    plt.ylabel("Value of distance metric")
    for change in changes:
        plt.axvline(change, color="red", label=f"change detected here at {change}")

    plt.plot(list(range(2 * n, N, shift)), KLs)
    plt.legend()
    return {"KL_values": KLs, "changes": changes}
