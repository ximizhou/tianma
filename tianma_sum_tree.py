import numpy as np

class PriorityTree:
    write_index = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0
        self.priority_exponent = 0.6  # prioritize exponent
        self.importance_sampling_exponent = 0.4  # importance-sampling exponent
        self.beta_increment = 0.001
        self.epsilon = 0.01  # small amount to avoid zero priority

    def _propagate(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index, sum_value):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if sum_value <= self.tree[left]:
            return self._retrieve(left, sum_value)
        else:
            return self._retrieve(right, sum_value - self.tree[left])

    def total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        index = self.write_index + self.capacity - 1
        self.data[self.write_index] = data
        self.update(index, priority)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, index, priority):
        priority = min(priority + self.epsilon, 1)  # Add epsilon to avoid zero priority and clip to max value
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def get(self, sum_value):
        index = self._retrieve(0, sum_value)
        data_index = index - self.capacity + 1

        self.importance_sampling_exponent = np.min([1., self.importance_sampling_exponent + self.beta_increment])  # increase beta
        priority = self.tree[index] ** self.importance_sampling_exponent
        importance_sampling_weight = (self.num_entries * priority) ** -self.importance_sampling_exponent  # importance sampling weight

        return (index, self.tree[index], self.data[data_index], importance_sampling_weight)

    def sample(self, batch_size):
        segment = self.total_priority() / batch_size
        batch = []
        indices = []
        priorities = []

        for i in range(batch_size):
            sum_value = np.random.uniform(segment * i, segment * (i + 1))
            index, priority, data, importance_sampling_weight = self.get(sum_value)
            batch.append(data)
            indices.append(index)
            priorities.append(importance_sampling_weight)

        return batch, indices, priorities

    def update_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            priority = (np.abs(error) + self.epsilon) ** self.priority_exponent
            self.update(index, priority)
