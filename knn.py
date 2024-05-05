import numpy as np

class DataPoint:
    def __init__(self, features, label):
        self.features = np.array(features)
        self.label = label

    def calculate_distance(self, other):
        return np.linalg.norm(self.features - other.features)

class TestPoint(DataPoint):
    pass

class DataSet:
    def __init__(self):
        self.points = []

    def add_point(self, data_point):
        self.points.append(data_point)

    def get_points(self):
        return self.points

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.training_data = DataSet()

    def train(self, data_set):
        self.training_data = data_set

    def predict(self, test_point):
        distances = []
        for point in self.training_data.get_points():
            dist = test_point.calculate_distance(point)
            distances.append((point, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]
        votes = {}
        for neighbor in neighbors:
            label = neighbor[0].label
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)

    def get_nearest_neighbour(self, test_point):
        nearest = None
        min_distance = float('inf')
        for point in self.training_data.get_points():
            dist = test_point.calculate_distance(point)
            if dist < min_distance:
                nearest = point
                min_distance = dist
        return nearest

# Example usage
data_set = DataSet()
data_set.add_point(DataPoint([1, 2], 'A'))
data_set.add_point(DataPoint([2, 3], 'B'))
data_set.add_point(DataPoint([3, 4], 'A'))

classifier = KNNClassifier(k=3)
classifier.train(data_set)

test_point = TestPoint([2, 2], None)
predicted_label = classifier.predict(test_point)
print("Predicted Label:", predicted_label)
