from typing import Any, List
from dataclasses import dataclass

import numpy as np
import random

@dataclass
class Experience:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class ExperienceBuffer:
    """
    O replay buffer existe para fornecer a média sobre amostras descorrelacionadas,
    que reduz a variância do gradiente.
    """
    def __init__(self, max_lenght=1024) -> None:
        self.max_lenght = max_lenght
        self.buffer: List[Experience] = []
        self.position = 0 # Added position for circular buffer

    def __len__(self):
        return len(self.buffer)

    def peak(self):
        if not self.buffer:
            raise ValueError("Buffer is empty")
        return self.buffer[self.position-1]

    def add(self, experience: Experience):
        if len(self.buffer) < self.max_lenght:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.max_lenght

    def sample(self, batch_size=32):
        # Ensure we don't sample more than available experiences
        actual_batch_size = min(batch_size, len(self.buffer))
        # Convert the buffer to a NumPy array before sampling
        # print(self.buffer, actual_batch_size)
        return random.sample(self.buffer, actual_batch_size)  # Use random.sample for sampling without replacement
        # return np.random.choice(self.buffer, actual_batch_size, replace=False) # Added replace=False for sampling without replacement
