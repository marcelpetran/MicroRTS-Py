import numpy as np
from collections import deque
from typing import Dict, Tuple
import random


class SumTree:
  """
  A SumTree is a Binary Tree where each parent node is the sum of its children.
  The leaves are the priorities of the transitions.
  """

  def __init__(self, capacity: int):
    self.capacity = capacity
    # The tree is 2*capacity - 1 nodes.
    # The first capacity-1 nodes are parents.
    # The last capacity nodes are the leaves (priorities).
    self.tree = np.zeros(2 * capacity - 1)
    # Store the actual transitions
    self.data = np.zeros(capacity, dtype=object)
    self.data_pointer = 0
    self.n_entries = 0

  def add(self, priority: float, data: object):
    """Add a new transition with a given priority."""
    # Store the data
    self.data[self.data_pointer] = data

    # Get the tree index for this data
    tree_idx = self.data_pointer + self.capacity - 1

    # Update the leaf with the new priority
    self.update(tree_idx, priority)

    # Move the pointer
    self.data_pointer = (self.data_pointer + 1) % self.capacity
    self.n_entries = min(self.n_entries + 1, self.capacity)

  def update(self, tree_idx: int, priority: float):
    """Update the priority of a node and propagate the change up the tree."""
    change = priority - self.tree[tree_idx]
    self.tree[tree_idx] = priority

    # Propagate the change up
    while tree_idx != 0:
      tree_idx = (tree_idx - 1) // 2
      self.tree[tree_idx] += change

  def get_leaf(self, s: float) -> tuple:
    """Find the leaf node for a given sum 's'."""
    idx = 0
    while True:
      left_child_idx = 2 * idx + 1
      right_child_idx = left_child_idx + 1

      # If we are at a leaf, stop
      if left_child_idx >= len(self.tree):
        leaf_idx = idx
        break
      else:
        # Follow the sum
        if s <= self.tree[left_child_idx]:
          idx = left_child_idx
        else:
          s -= self.tree[left_child_idx]
          idx = right_child_idx

    data_idx = leaf_idx - self.capacity + 1
    return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

  @property
  def total_priority(self) -> float:
    """Get the total sum of all priorities (the root node)."""
    return self.tree[0]

  def __len__(self):
    return self.n_entries


class PrioritizedReplayBuffer:
  """
  The PER Buffer. It uses a SumTree to manage priorities.
  """
  # Epsilon: a small value to ensure no transition has 0 priority
  eps = 1e-6
  # Alpha: determines how much prioritization is used (0=uniform, 1=full)
  alpha = 0.6
  # Beta: Importance-sampling correction (starts at 0.4, anneals to 1.0)
  beta_start = 0.4
  beta_inc = 1e-4

  def __init__(self, capacity: int):
    self.tree = SumTree(capacity)
    self.capacity = capacity
    self.beta = self.beta_start
    # We must store the max priority for new samples
    self.max_priority = 1.0

  def push(self, item: Dict):
    """
    Push a new item. New items are given the *maximum* priority
    to ensure they are trained on at least once.
    """
    self.tree.add(self.max_priority, item)

  def sample(self, batch_size: int) -> tuple:
    """Sample a batch, returning transitions, IS weights, and tree indices."""
    batch = []
    is_weights = np.zeros(batch_size, dtype=np.float32)
    tree_indices = np.zeros(batch_size, dtype=int)

    # Segment the total priority into 'batch_size' chunks
    segment_size = self.tree.total_priority / batch_size

    # Anneal beta
    self.beta = np.min([1., self.beta + self.beta_inc])

    # Calculate min probability for IS weight calculation
    min_prob = np.min(
      self.tree.tree[-self.capacity:]) / self.tree.total_priority

    for i in range(batch_size):
      # Sample a value from each segment
      a = segment_size * i
      b = segment_size * (i + 1)
      s = random.uniform(a, b)

      # Get the leaf corresponding to that value
      (tree_idx, priority, data) = self.tree.get_leaf(s)

      # Calculate sampling probability and IS weight
      prob = priority / self.tree.total_priority
      is_weights[i] = (prob / min_prob) ** (-self.beta)

      batch.append(data)
      tree_indices[i] = tree_idx

    return batch, is_weights, tree_indices

  def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
    """
    Update the priorities of sampled transitions.
    """
    # Convert TD errors to priorities (add epsilon, raise to alpha)
    priorities = (np.abs(td_errors) + self.eps) ** self.alpha

    for idx, p in zip(tree_indices, priorities):
      self.tree.update(idx, p)

    # Update the max priority
    self.max_priority = max(self.max_priority, np.max(priorities))

  def __len__(self):
    return len(self.tree)
