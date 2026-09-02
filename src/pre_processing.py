import torch
import numpy as np

from collections import deque
from PIL import Image, ImageOps

# import cv2
# from torchvision import transforms


class ImagePreprocessor:
    def __init__(self, m=4, frame_size=84):
        """
        Image frame preprocessing.

        Args:
            m: Number of frames to stack (default: 4)
            frame_size: Target size for rescaled frames (default: 84x84)
        """
        self.m = m
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=2)  # Buffer for max pooling over 2 frames
        self.state_buffer = deque(maxlen=m)  # Buffer for stacking m frames

    def reset(self):
        """Reset the buffers (call at start of new episode)"""
        self.frame_buffer.clear()
        self.state_buffer.clear()

    def preprocess_frame(self, _frame):
        """
        Preprocess a single frame:
        1. Max pooling over current and previous frame (removes flickering)
        2. Convert RGB to grayscale (Y channel/luminance)
        3. Resize to 84x84

        Args:
            frame: numpy array of shape (H, W, 3) with RGB values [0-255]

        Returns:
            Preprocessed frame of shape (84, 84) with values [0-255]
        """
        # Add current frame to buffer
        frame = np.transpose(_frame, (2, 0, 1))
        self.frame_buffer.append(frame)

        # Max pooling over frames (removes flickering)
        if len(self.frame_buffer) == 2:
            max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        else:
            max_frame = frame

        # Convert RGB to grayscale (luminance)
        # Using the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
        # gray_frame = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        gray_frame = max_frame[1]

        # Resize to 84x84
        # resized_frame = cv2.resize(gray_frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
        with Image.fromarray(gray_frame) as img:
            img = ImageOps.fit(img, (self.frame_size, self.frame_size), Image.LANCZOS)
            resized_frame = np.array(img)

        return resized_frame

    def get_state(self, frame):
        """
        Get the stacked state (phi) from a new frame.

        Args:
            frame: Raw Atari frame of shape (H, W, 3)

        Returns:
            State tensor of shape (m, 84, 84) or (4, 84, 84) by default
            For the first m frames, repeats the frame to fill the buffer
        """
        # Preprocess the frame
        processed_frame = self.preprocess_frame(frame)

        # Add to state buffer
        self.state_buffer.append(processed_frame)

        # If we don't have enough frames yet, repeat the current frame
        while len(self.state_buffer) < self.m:
            self.state_buffer.append(processed_frame)

        # Stack the m most recent frames
        state = np.stack(self.state_buffer, axis=0)

        return state

    def get_state_tensor(self, frame):
        """
        Get state as PyTorch tensor ready for the network.

        Args:
            frame: Raw Atari frame of shape (H, W, 3)

        Returns:
            State tensor of shape (1, m, 84, 84) with values normalized to [0, 1]
        """
        state = self.get_state(frame)

        # Convert to float and normalize to [0, 1]
        state = state.astype(np.float32) / 255.0

        # Convert to PyTorch tensor (m, 84, 84) -> no batch is the standard for the network
        # batch dimension is added later when forwarding
        state_tensor = torch.from_numpy(state)

        return state_tensor


# Example usage:
if __name__ == "__main__":
    # Simulate Atari frames (210x160x3)
    preprocessor = ImagePreprocessor(m=4, frame_size=84)

    # Start of episode
    preprocessor.reset()

    # Process some frames
    for i in range(10):
        # Simulate a frame (replace with actual Atari frame)
        fake_frame = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)

        # Get the stacked state
        state = preprocessor.get_state(fake_frame)
        print(f"Frame {i}: State shape = {state.shape}")  # (4, 84, 84)

        # Or get as PyTorch tensor
        state_tensor = preprocessor.get_state_tensor(fake_frame)
        print(f"Frame {i}: Tensor shape = {state_tensor.shape}")  # (1, 4, 84, 84)
