class UnionFind(object):
    """
    Disjoint set Union and Find for Boruvka's algorithm
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def __len__(self):
        return len(self.parent)

    def make_set(self, item):
        if item in self.parent:
            return self.find(item)

        self.parent[item] = item
        self.rank[item] = 0
        return item

    def find(self, item):
        if item not in self.parent:
            return self.make_set(item)
        if item != self.parent[item]:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 == root2:
            return root1

        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            return root1

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            return root2

        if self.rank[root1] == self.rank[root2]:
            self.rank[root1] += 1
            self.parent[root2] = root1
            return root1
