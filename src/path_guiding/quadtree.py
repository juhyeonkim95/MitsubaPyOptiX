import numpy as np
import open3d as o3d

from collections import deque
import math


class DTree:
    def __init__(self, uuid, value_array, index_array, rank_array):
        self.index_array = index_array
        self.rank_array = rank_array
        self.value_array = value_array
        self.max_length = len(index_array)
        self.current_size = 1
        self.select_array = np.zeros_like(self.index_array, dtype=np.uint32)
        self.depth_array = np.zeros_like(self.index_array, dtype=np.uint32)
        self.uuid = uuid


        # x = 0
        # s = 1
        # while x + s < self.max_length:
        #     x += s
        #     s *= 4
        # self.index_array[0:x] = 1
        # self.build_rank_array()
    # def build(self, threshold):
    #     node_q = deque()
    #     node_q.append(0)
    #     total_sum = self.value_array[0]
    #
    #     while True:
    #         node = node_q.pop()
    #         value = self.value_array[node]
    #         if value / total_sum > threshold and self.current_size + 4 < self.max_length:
    #             self.index_array[node] = 1
    #             self.current_size += 4
    #             for i in range(4):
    #                 child_idx = self.child(node, i)
    #                 self.value_array[child_idx] = value / 4
    #                 node_q.append(child_idx)

    def build_rank_array(self):
        sum = 0
        for i in range(self.current_size):
            self.rank_array[i] = sum
            sum += self.index_array[i]

        for i in range(self.current_size):
            if self.index_array[i] == 1:
                for j in range(4):
                    child_j = self.child(i, j)
                    self.select_array[child_j] = i

    def child(self, idx, child_idx):
        return 4 * (self.rank_array[idx]) + child_idx + 1

    def parent(self, child_idx):
        return self.select_array[child_idx]

    def update_parent_radiance(self):
        for i in reversed(range(self.current_size)):
            if i == 0:
                continue
            parent_id = self.parent(i)
            assert self.index_array[parent_id] == 1
            self.value_array[parent_id] += self.value_array[i] * 0.25

    def get_total_irradiance(self):
        irradiance = 0
        area = 0
        for i in range(self.current_size):
            if self.index_array[i] == 0:
                irradiance += self.value_array[i] * pow(0.25, self.depth_array[i])
                area += pow(0.25, self.depth_array[i])

        assert area == 1.0
        return irradiance

        # for i in reversed(range(self.current_size)):
        #     if i == 0:
        #         break
        #     parent_idx = self.parent(i)
        #     self.value_array[parent_idx] += self.value_array[i]
        #     self.visit_count_array[parent_idx] += self.visit_count_array[i]

    def visualize_quadtree(self, image_size=512):
        q = deque()
        q.append(0)
        pose_sizes = deque()
        pose_sizes.append(((0, 0), 1))
        result_array = np.zeros((image_size, image_size))

        while len(q) > 0:
            node = q.pop()
            origin, size = pose_sizes.pop()
            if self.index_array[node] == 1:
                for i in range(4):
                    q.append(self.child(node, i))
                    nx = origin[0] + 0.5 * (size if i >= 2 else 0)
                    ny = origin[1] + 0.5 * (size if i % 2 == 1 else 0)
                    pose_sizes.append(((nx, ny), size * 0.5))
            else:
                sx = int(image_size * origin[0])
                sy = int(image_size * origin[1])
                s = int(image_size * size)
                a = self.value_array[node] / (np.max(self.value_array) + 1e-6)
                result_array[sx:sx+s, sy:sy+s].fill(a)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(result_array)
        plt.show()

    def is_leaf_node(self, m):
        return self.index_array[m] == 0

    def print(self):
        print("Value array", self.value_array[0:self.current_size])
        print("Index array", self.index_array[0:self.current_size])
        print("Rank array", self.rank_array[0:self.current_size])
        # print("Visit array", self.visit_count_array[0:self.current_size])
        print("Depth array", self.depth_array[0:self.current_size])
        print("Select array", self.select_array[0:self.current_size])

    def update(self, threshold=0.01):
        if self.current_size == self.max_length:
            return

        sum = self.get_total_irradiance()

        q = deque()
        q.append((0, self.value_array[0], 0))
        index_array = deque()
        value_array = deque()
        depth_array = deque()
        current_size = self.current_size
        while len(q) > 0:
            node, val, depth = q.popleft()
            depth_array.append(depth)
            if node < self.current_size:
                # internal node
                if self.index_array[node] == 1:
                    index_array.append(1)
                    value_array.append(0)
                    for i in range(4):
                        child_id = self.child(node, i)
                        q.append((child_id, self.value_array[node] / 4, depth + 1))
                # leaf node
                else:
                    if self.value_array[node] * pow(0.25, depth) >= threshold * sum \
                            and current_size + 4 <= self.max_length:
                        index_array.append(1)
                        for i in range(4):
                            q.append((current_size + i, self.value_array[node] / 4, depth + 1))
                        value_array.append(0)
                        current_size += 4
                    else:
                        index_array.append(0)
                        value_array.append(self.value_array[node])
            else:
                index_array.append(0)
                value_array.append(val)

        self.current_size = len(index_array)
        assert current_size == self.current_size
        #self.index_array[0] = 1
        self.index_array[0:self.current_size] = np.array(index_array)
        self.value_array[0:self.current_size] = np.array(value_array)
        self.depth_array[0:self.current_size] = np.array(depth_array)
        self.build_rank_array()
        self.update_parent_radiance()

        # if self.uuid == 188:
        #     print("After")
        #     self.print()



    def update_old(self, threshold=0.01):
        # self.update_parent_q_value()

        current_size = self.current_size
        sum = np.sum(self.value_array[0:self.current_size])
        # threshold = self.value_array[0] * threshold

        for i in range(current_size):
            if self.is_leaf_node(i):
                if self.value_array[i] > threshold * sum:
                    self.split(i)

        self.build_rank_array()

    def split(self, idx):
        if self.current_size + 4 <= self.max_length:
            # make idx inner node
            self.index_array[idx] = 1
            self.current_size += 4
        # value = self.value_array[idx] / 4
        # for i in range(4):
        #     child_idx = self.child(idx, i)
        #     self.value_array[child_idx] = value


class Octree:
    def __init__(self, index_array):
        def cumsum(array):
            rank_array = np.cumsum(array, dtype=np.uint32)
            rank_array = np.insert(rank_array, 0, 0)[0:-1]
            return rank_array
        self.index_array = index_array
        rank_array = cumsum(index_array)
        print("Index array", index_array[0:20])
        print("Rank array", rank_array[0:20])

        self.rank_array = rank_array
        self.total_size = len(index_array)
        self.node_number = (self.total_size - 1) // 8

        leaf_array = np.zeros(self.node_number, dtype=np.uint32)
        for i in range(self.node_number):
            any_leaf = False
            for j in range(8):
                child_idx = 8 * self.rank_array[i] + j
                if self.index_array[child_idx] == 0:
                    any_leaf = True
            if any_leaf:
                leaf_array[i] = 1

        self.leaf_node_number = int(np.sum(leaf_array))
        self.rank_leaf_array = cumsum(leaf_array)

        assert (self.total_size - 1) % 8 == 0
        print("Octree created with total %d nodes" % self.node_number)
        print("Total node %d, leaf %d" % (self.node_number, self.leaf_node_number))
        self.test()

    def get_index(self, p):
        idx = 0
        a = np.copy(p)
        while True:
            x = a[0] >= 0.5
            y = a[1] >= 0.5
            z = a[2] >= 0.5
            child_idx = z << 2 | y << 1 | x
            a[0] = 2 * a[0] if not x else 2 * a[0] - 1
            a[1] = 2 * a[1] if not y else 2 * a[1] - 1
            a[2] = 2 * a[2] if not z else 2 * a[2] - 1
            child = 8 * self.rank_array[idx] + child_idx + 1
            if self.index_array[child] == 0:
                break
            idx = child
        return self.rank_array[idx]

    def test(self):
        for _ in range(100):
            idx = self.get_index(np.random.rand(3))
            assert 0 <= idx < self.node_number




    # def update(self, k, c=12000):
    #     '''
    #     Update spatial binary tree with regard to visited count.
    #     split if (# of sample) > c * sqrt(2 ^ k)
    #     :return: None
    #     '''
    #     threshold = int(c * math.pow(2, k / 2))
    #
    #     new_index_array = []
    #     new_visit_array = []
    #
    #     # append (node index, visit count, )
    #     q = deque()
    #     initial_pair = (0, self.visit_count_array[0])
    #     q.append(initial_pair)
    #
    #     while len(q) > 0:
    #         node, visited_count, depth, parent_node = q.popleft()
    #
    #         # existing node
    #         if node < self.size:
    #             visited_count = self.visit_count_array[node]
    #             # leaf node
    #             if self.is_leaf(node):
    #                 if visited_count > threshold:
    #                     new_index_array.append(1)
    #                 else:
    #                     new_index_array.append(0)
    #             # internal node
    #             else:
    #                 new_index_array.append(1)
    #         # newly added node
    #         else:
    #             new_index_array.append(0)


    #
    # def build_rank_array(self):
    #     sum = 0
    #     for i in range(self.total_size):
    #         self.rank_array[i] = sum
    #         sum += self.index_array[i]
    #
    #     for i in range(self.total_size):
    #         if self.index_array[i] == 1:
    #             for j in range(2):
    #                 child_j = self.child(i, j)
    #                 self.select_array[child_j] = i
    #
    # def child(self, idx, child_idx):
    #     return 2 * (self.rank_array[idx]) + child_idx + 1
    #
    # def update(self, visit_array, k, c):
    #     q = deque()
    #     q.append((0, visit_array[0], 0))
    #
    #     threshold = int(c * math.pow(2, k / 2))
    #     new_index_array = []
    #     new_visit_array = []
    #     new_axis_array = []
    #     dtree_index_array = []
    #     current_size = self.total_size
    #
    #     while len(q) > 0:
    #         node, visited_count, depth, parent_node = q.popleft()
    #         if node < self.total_size:
    #             visited_count = visit_array[node]
    #             # leaf node
    #             if self.is_leaf(node):
    #                 if visited_count > threshold:
    #                     new_index_array.append(1)
    #                     q.append((current_size + 0, visited_count / 2, depth + 1, node))
    #                     q.append((current_size + 1, visited_count / 2, depth + 1, node))
    #                     current_size += 2
    #                 else:
    #                     new_index_array.append(0)
    #                     new_visit_array.append(visited_count)
    #                     dtree_index_array.append(node)
    #             # internal node
    #             else:
    #                 new_index_array.append(1)
    #                 new_visit_array.append(visited_count)
    #                 dtree_index_array.append(node)
    #         else:
    #             new_index_array.append(0)
    #             new_visit_array.append(visited_count)
    #             dtree_index_array.append(parent_node)


class SDTree:
    def __init__(self, octree, n_max_dir=1000):
        self.octree = octree
        self.q_table = np.zeros((octree.node_number, n_max_dir), dtype=np.float32)
        self.index_array = np.zeros((octree.node_number, n_max_dir), dtype=np.float32)

        self.d_trees = []
        for i in range(octree.node_number):
            self.d_trees.append(DTree(self.q_table[i], self.index_array[i]))

    def update(self):
        for dtree in self.d_trees:
            dtree.update()


def traverse(root):
    queue = [root]

    while len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node, o3d.geometry.OctreeInternalNode):
            for child in node.children:
                traverse(child)


class OctreeNode:
    def __init__(self):
        self.children = [None] * 8


def octree_to_index_array(octree):
    q = deque()
    index_array = deque()
    index_array.append(1)
    q.append(octree.root_node)
    while len(q) > 0:
        node = q.popleft()
        if isinstance(node, o3d.geometry.OctreeInternalNode):
            for child in node.children:
                if child is None:
                    index_array.append(0)
                else:
                    index_array.append(1)
                    q.append(child)
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            for _ in range(8):
                index_array.append(0)

    index_array_np = np.array(index_array, dtype=np.uint32)
    return index_array_np


def f_traverse_build_octree(node, node_info):
    early_stop = False
    octree_nodes = OctreeNode()

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                octree_nodes.children[node_info.child_index]
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop