class QuadTreeNode:
    def __init__(self):

class DTree:
    def __init__(self):
        self.nodes = None

class DTreeWrapper:
    def __init__(self):
        self.building = None
        self.sampling = None

class STreeNode:
    def __init__(self):
        self.children = [0, 0]
        self.is_leaf = True
        self.axis = 0
        self.dtree = None

class STree:
    def __init__(self, aabb):
        self.nodes = []
        self.aabb = aabb

    def clear(self):
        self.nodes.clear()

    def subdivide_all(self):
        for node in self.nodes:
            if node.is_leaf:
                self.subdivide(node)

    def subdivide(self, node):
