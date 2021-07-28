import numpy as np
from pyoptix import Buffer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from collections import deque
import copy


class Box:
    def __init__(self):
        self.s = np.array([0, 0, 0], dtype=np.float32)
        self.e = np.array([1, 1, 1], dtype=np.float32)

    def __str__(self):
        return "Min %s Max %s" % (str(self.s), str(self.e))


def plot_cube_box(ax, min, max):
    c1 = (min[0], min[1], min[2])
    c2 = (max[0], min[1], min[2])
    c3 = (min[0], max[1], min[2])
    c4 = (min[0], min[1], max[2])
    plot_cube(ax, [c1, c2, c3, c4])


def plot_cube(ax, cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0))
    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    # ax.scatter(points[:,0], points[:,1], points[:,2], s=0)
    #plt.show()
    # ax.set_aspect('equal')


class SpatialAdaptiveBinaryTree:
    """
    Implement spatial binary tree in SOA (Structure of Arrays) style
    """
    def __init__(self, max_size=10000, initial_subdivision=0):
        self.max_size = max_size
        self.before_size = 1
        self.size = 1
        self.visit_count_array = np.zeros((max_size,), dtype=np.uint32)
        self.axis_array = np.zeros((max_size,), dtype=np.uint32)

        # store only first child index, assume children are in consecutive sequence.
        self.child_array = np.zeros((max_size, ), dtype=np.uint32)
        self.parent_array = np.zeros((max_size, ), dtype=np.uint32)

        for i in range(initial_subdivision):
            self.subdivide_all()

        print("Initial Size", self.size)

    def position_to_index(self, _p):
        idx = 0
        p = np.copy(_p)
        while True:
            if self.is_leaf(idx):
                break

            axis = self.axis_array[self.child_array[idx]]
            if p[axis] < 0.5:
                p[axis] = 2 * p[axis]
                child_local_idx = 0
            else:
                p[axis] = 2 * p[axis] - 1
                child_local_idx = 1
            idx = self.child_array[idx] + child_local_idx
        return idx

    def is_leaf(self, node):
        return self.child_array[node] == 0

    def subdivide(self, node):
        if self.size + 2 > self.max_size:
            return

        self.child_array[node] = self.size
        for i in range(2):
            idx = self.size + i
            self.axis_array[idx] = (self.axis_array[node] + 1) % 3
            self.parent_array[idx] = node
            self.visit_count_array[idx] = self.visit_count_array[node] // 2
        self.visit_count_array[node] = 0

        self.size += 2

    def subdivide_all(self):
        self.before_size = self.size
        for node in range(self.before_size):
            if self.size >= self.max_size:
                break
            if self.is_leaf(node):
                self.subdivide(node)

    def refine(self, threshold, invalid_rates=None, invalid_rate_threshold=0):
        """
        Refine spatial binary tree with regard to visited count.
        split if (# of sample) > threshold
        :return: None
        """

        node_dequeue = deque()
        node_dequeue.append(0)
        pairs = []

        while len(node_dequeue) > 0:
            if self.size >= self.max_size:
                break

            node = node_dequeue.popleft()
            if self.is_leaf(node):
                visited_count = self.visit_count_array[node]
                do_split = (visited_count > threshold) if threshold > 0 else False
                # do_split = do_split or (invalid_rates[node] > invalid_rate_threshold)
                if do_split:
                    self.subdivide(node)

                    child_idx = self.child_array[node]
                    pairs.append((child_idx, node))
                    pairs.append((child_idx + 1, node))

            if not self.is_leaf(node):
                child_idx = self.child_array[node]
                node_dequeue.append(child_idx)
                node_dequeue.append(child_idx + 1)

        return pairs

    def refine_single(self, threshold):
        '''
        Refine spatial binary tree with regard to visited count.
        split if (# of sample) > threshold
        :return: None
        '''

        self.before_size = self.size

        for node in range(self.before_size):
            if self.size >= self.max_size:
                break
            if self.is_leaf(node):
                visited_count = self.visit_count_array[node]
                if visited_count > threshold:
                    self.subdivide(node)
        pairs = []
        for idx in range(self.before_size, self.size, 1):
            pairs.append((idx, self.parent_array[idx]))
        return pairs

    def print(self):
        print("child_array", self.child_array[0:self.size])
        print("parent_array", self.parent_array[0:self.size])
        print("axis_array", self.axis_array[0:self.size])
        print("visit_count_array", self.visit_count_array[0:self.size])

    def copy_to_context(self, context):
        context["stree_child_array"].copy_from_array(self.child_array)
        context["stree_parent_array"].copy_from_array(self.parent_array)
        context["stree_axis_array"].copy_from_array(self.axis_array)
        context["stree_visit_count"].copy_from_array(self.visit_count_array)

    def copy_from_context(self, context):
        context["stree_child_array"].copy_to_array(self.child_array)
        context["stree_parent_array"].copy_to_array(self.parent_array)
        context["stree_axis_array"].copy_to_array(self.axis_array)
        context["stree_visit_count"].copy_to_array(self.visit_count_array)

    def create_buffer_to_context(self, context):
        context["stree_child_array"] = Buffer.from_array(self.child_array, buffer_type='io', drop_last_dim=False)
        context["stree_parent_array"] = Buffer.from_array(self.parent_array, buffer_type='io', drop_last_dim=False)
        context["stree_axis_array"] = Buffer.from_array(self.axis_array, buffer_type='io', drop_last_dim=False)
        context["stree_visit_count"] = Buffer.from_array(self.visit_count_array, buffer_type='io', drop_last_dim=False)
        context['stree_size'] = Buffer.from_array(np.array([self.size, 0], dtype=np.uint32), buffer_type='io', drop_last_dim=False)

    def visualize(self, invalid_rates=None):
        boxes = []
        for i in range(self.size):
            if i == 0:
                boxes.append(Box())
                continue
            parent_index = self.parent_array[i]
            parent_box = boxes[parent_index]
            child_box = copy.deepcopy(parent_box)
            axis = self.axis_array[i]
            child_idx = i - self.child_array[parent_index]
            if child_idx == 0:
                child_box.e[axis] = (parent_box.s[axis] + parent_box.e[axis]) / 2.0
            else:
                child_box.s[axis] = (parent_box.s[axis] + parent_box.e[axis]) / 2.0

            boxes.append(child_box)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("auto")
        ax.set_autoscale_on(True)
        points = []
        colors = []
        scales = []

        red = np.array([1, 0, 0], dtype=np.float32)
        green = np.array([0, 1, 0], dtype=np.float32)

        for i in range(self.size):
            if self.is_leaf(i):
                points.append((boxes[i].s + boxes[i].e) * 0.5)

                if invalid_rates is not None:
                    invalid_rate = invalid_rates[i]
                    max_rate = 1.0
                    min_rate = 0.0
                    invalid_rate = (invalid_rate - min_rate) / (max_rate - min_rate)
                    invalid_rate = np.clip(invalid_rate, 0, 1)
                    color = red * invalid_rate + green * (1-invalid_rate)
                else:
                    color = red
                colors.append(color)
                scale = self.visit_count_array[i] / np.max(self.visit_count_array) * 30
                scales.append(scale)

                #scatter_cube(ax, boxes[i].s, boxes[i].e)
                #plot_cube_box(ax, boxes[i].s, boxes[i].e)
        points = np.array(points).transpose()
        # plt.style.use('dark_background')
        ax.set_axis_off()
        ax.scatter(points[0], points[1], points[2], s=scales, c=colors)
        ax.set_xlim3d(0, 1)
        ax.set_ylim3d(0, 1)
        ax.set_zlim3d(0, 1)

        plt.show()
