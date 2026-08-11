from typing import Deque, Dict, List, Tuple, Optional
from collections import deque
import random
import torch


class ReplayBuffer:
  """FIFO replay as a list-backed circular buffer: O(1) push, O(k) sample."""

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf = [None] * capacity
    self.ptr = 0        # next slot to overwrite
    self.size = 0

  def push(self, item):
    self.buf[self.ptr] = item
    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size: int):
    idxs = random.sample(range(self.size), batch_size)
    return [self.buf[i] for i in idxs]

  def __len__(self):
    return self.size


class ReservoirBuffer:
  """Reservoir Sampler for SL."""

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf = []
    self.n_seen = 0

  def push(self, item: Dict):
    if len(self.buf) < self.capacity:
      self.buf.append(item)
    else:
      j = random.randint(0, self.n_seen)
      if j < self.capacity:
        self.buf[j] = item
    self.n_seen += 1

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)


class ReservoirBufferExponentialAveraging:
  """Reservoir sampler biased toward recently inserted items."""

  def __init__(self, capacity: int, decay_tau: float = None):
    self.capacity = capacity
    self.buf = [None] * capacity
    self.arrival = torch.zeros(capacity, dtype=torch.float32)
    self.size = 0
    self.n_seen = 0
    self.decay_tau = decay_tau

  def push(self, item):
    self.n_seen += 1
    if self.size < self.capacity:
      slot = self.size
      self.size += 1
      self.buf[slot] = item
      self.arrival[slot] = float(self.n_seen)
    else:
      j = random.randint(1, self.n_seen)
      if j <= self.capacity:
        self.buf[j - 1] = item
        self.arrival[j - 1] = float(self.n_seen)

  def sample(self, batch_size: int):
    n = self.size
    if n <= batch_size:
      return [self.buf[i] for i in range(n)]
    if self.decay_tau is None:
      # linear recency
      weights = self.arrival[:n]
    else:
      # true exponential
      weights = torch.exp(
        (self.arrival[:n] - self.n_seen) / self.decay_tau)
    idx = torch.multinomial(weights, batch_size, replacement=False)
    return [self.buf[i] for i in idx.tolist()]

  def __len__(self):
    return self.size
