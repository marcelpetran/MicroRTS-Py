from typing import Deque, Dict, List, Tuple, Optional
from collections import deque
import random

class ReplayBuffer:
  """Standard FIFO buffer for Q-learning (requires recent data)."""

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf: Deque[Dict] = deque(maxlen=capacity)

  def push(self, item: Dict):
    self.buf.append(item)

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)


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
  
class ReservoirBufferExonentialAveraging:
  """Reservoir Sampler for SL with Exonential Averaging."""

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf = []
    self.counts = []
    self.n_seen = 1

  def push(self, item: Dict):
    if len(self.buf) < self.capacity:
      self.buf.append(item)
      self.counts.append(self.n_seen)
    else:
      j = random.randint(0, self.n_seen)
      if j < self.capacity:
        self.buf[j] = item
        self.counts[j] = self.n_seen
    self.n_seen += 1

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, k=batch_size, counts=self.counts)

  def __len__(self):
    return len(self.buf)