import numpy as np
from typing import List, Tuple, Optional

class MapCamera:
    """
    A class to manage map visualization with camera tracking, worm detection, and boundary management.
    """
    
    # Color constants
    RED = np.array([0, 0, 255])
    BLUE = np.array([255, 0, 0]) 
    WHITE = np.array([255, 255, 255])
    
    def __init__(self, map_size: int = 700, center: int = 350):
        """
        Initialize the MapCamera.
        
        Args:
            map_size: Size of the map
            center: Center position of the map
        """
        self.map_size = map_size
        self.center = center
        self.map_camera = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        self.position_history: List[List[int]] = []
        self.shift_x = 0
        self.shift_y = 0
        self.next_flag = False
        
        # Rectangle drawing parameters
        self.rect_half_height = 40
        self.rect_half_width = self.rect_half_height
        
        # Boundary parameters
        self.boundary_low = 40
        self.boundary_high = map_size - self.boundary_low
    
    def _is_color_match(self, pixel: np.ndarray, color: np.ndarray) -> bool:
        """Check if a pixel matches a specific color."""
        return np.array_equal(pixel, color)
    
    def _get_not_red_mask(self, region: np.ndarray) -> np.ndarray:
        """Get a mask for pixels that are not red."""
        return ~np.all(region == self.RED, axis=-1)
    
    def draw_worm(self, x: int, y: int, color: np.ndarray = None) -> None:
        """
        Draw a worm shape at the specified position.
        
        Args:
            x, y: Position coordinates
            color: Color to draw with (defaults to RED)
        """
        if color is None:
            color = self.RED
            
        # Main body
        self.map_camera[x, y-4:y+4] = color
        self.map_camera[x-1, y-2:y+2] = color
        
        # Extensions
        self.map_camera[x+1, y+3:y+4] = color
        self.map_camera[x+1, y-4:y-3] = color
        self.map_camera[x+2, y+4] = color
        self.map_camera[x+2, y-4] = color
        
        # Additional details
        self.map_camera[x+1:x+2, y+5] = color
        self.map_camera[x+2:x+3, y+6] = color
    
    def draw_rectangle(self, x: int, y: int, color: np.ndarray = None) -> None:
        """
        Draw a filled rectangle at the specified position, avoiding red pixels.
        
        Args:
            x, y: Center position coordinates
            color: Color to draw with (defaults to WHITE)
        """
        if color is None:
            color = self.WHITE
            
        x1 = x - self.rect_half_width
        x2 = x + self.rect_half_width + 1
        y1 = y - self.rect_half_height
        y2 = y + self.rect_half_height + 1
        
        # Ensure bounds are within map
        x1 = max(0, x1)
        x2 = min(self.map_size, x2)
        y1 = max(0, y1)
        y2 = min(self.map_size, y2)
        
        subregion = self.map_camera[x1:x2, y1:y2]
        not_red_mask = self._get_not_red_mask(subregion)
        subregion[not_red_mask] = color
        self.map_camera[x1:x2, y1:y2] = subregion
    
    def draw_border(self, x: int, y: int, color: np.ndarray = None) -> None:
        """
        Draw a border rectangle at the specified position, avoiding red pixels.
        
        Args:
            x, y: Center position coordinates
            color: Color to draw with (defaults to WHITE)
        """
        if color is None:
            color = self.WHITE
            
        x1 = x - self.rect_half_width
        x2 = x + self.rect_half_width + 1
        y1 = y - self.rect_half_height
        y2 = y + self.rect_half_height + 1
        
        # Ensure bounds are within map
        x1 = max(0, x1)
        x2 = min(self.map_size, x2)
        y1 = max(0, y1)
        y2 = min(self.map_size, y2)
        
        # Draw border edges
        edges = [
            (slice(x1, x1+1), slice(y1, y2)),  # Left edge
            (slice(x2-1, x2), slice(y1, y2)),  # Right edge
            (slice(x1, x2), slice(y1, y1+1)),  # Top edge
            (slice(x1, x2), slice(y2-1, y2))   # Bottom edge
        ]
        
        for x_slice, y_slice in edges:
            if x_slice.start < self.map_size and y_slice.start < self.map_size:
                subregion = self.map_camera[x_slice, y_slice]
                not_red_mask = self._get_not_red_mask(subregion)
                subregion[not_red_mask] = color
                self.map_camera[x_slice, y_slice] = subregion
    
    def _get_position_bounds(self) -> Tuple[int, int, int, int]:
        """Get the bounds of all positions in history."""
        if not self.position_history:
            return self.center, self.center, self.center, self.center
            
        positions = np.array(self.position_history)
        return (
            int(np.max(positions[:, 0])),  # max_x
            int(np.min(positions[:, 0])),  # min_x
            int(np.max(positions[:, 1])),  # max_y
            int(np.min(positions[:, 1]))   # min_y
        )
    
    def check_and_adjust_boundaries(self, new_pos_x: int, new_pos_y: int) -> Tuple[int, int]:
        """
        Check if the new position requires boundary adjustment and perform translation.
        
        Args:
            new_pos_x, new_pos_y: New position coordinates
            
        Returns:
            Tuple of (translate_x, translate_y) applied
        """
        max_x, min_x, max_y, min_y = self._get_position_bounds()
        
        translate_x = 0
        translate_y = 0
        
        # Check X boundaries
        if new_pos_x >= self.boundary_high:
            translate_x = (self.boundary_low - min_x) // 2
        elif new_pos_x <= self.boundary_low:
            translate_x = (self.boundary_high - max_x) // 2
        
        # Check Y boundaries
        if new_pos_y >= self.boundary_high:
            translate_y = (self.boundary_low - min_y) // 2
        elif new_pos_y <= self.boundary_low:
            translate_y = (self.boundary_high - max_y) // 2
        
        # Apply translation if needed
        if translate_x != 0 or translate_y != 0:
            self.map_camera = np.roll(self.map_camera, shift=(translate_x, translate_y), axis=(0, 1))
            self.position_history = [[x + translate_x, y + translate_y] for x, y in self.position_history]
        
        return translate_x, translate_y
    
    def world_to_map_coordinates(self, pos_x: float, pos_y: float, init_pos_x: float, init_pos_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to map coordinates.
        
        Args:
            pos_x, pos_y: Current world position
            init_pos_x, init_pos_y: Initial world position
            
        Returns:
            Tuple of (map_x, map_y) coordinates
        """
        map_x = int(self.center + (pos_x - init_pos_x) / 40) + self.shift_x
        map_y = int(self.center - (pos_y - init_pos_y) / 40) + self.shift_y
        return map_x, map_y
    
    def replace_blue_with_white(self) -> None:
        """Replace all blue pixels with white pixels."""
        blue_mask = np.all(self.map_camera == self.BLUE, axis=-1)
        self.map_camera[blue_mask] = self.WHITE
    
    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """Get the last position from history."""
        if not self.position_history:
            return None
        return tuple(self.position_history[-1])
    
    def update(self, pos_x: float, pos_y: float, init_pos_x: float, init_pos_y: float, 
               find_worm: bool = False) -> Tuple[int, int, bool]:
        """
        Main update method that handles all map updates.
        
        Args:
            pos_x, pos_y: Current world position
            init_pos_x, init_pos_y: Initial world position
            find_worm: Whether a worm was found
            
        Returns:
            Tuple of (total_shift_x, total_shift_y, next_flag)
        """
        # Convert world coordinates to map coordinates
        new_pos_x, new_pos_y = self.world_to_map_coordinates(pos_x, pos_y, init_pos_x, init_pos_y)
        
        # Get last position
        last_pos = self.get_last_position()
        last_pos_x, last_pos_y = last_pos if last_pos else (0, 0)
        
        if find_worm:
            # Draw worm and reset translation
            self.draw_worm(new_pos_x, new_pos_y, self.RED)
            translate_x = translate_y = 0
            
        elif (new_pos_x, new_pos_y) != (last_pos_x, last_pos_y):
            # New position detected
            translate_x, translate_y = self.check_and_adjust_boundaries(new_pos_x, new_pos_y)
            
            # Update position after translation
            new_pos_x += translate_x
            new_pos_y += translate_y
            self.position_history.append([new_pos_x, new_pos_y])
            
            # Handle previous position marking
            if not self.next_flag:
                # Draw white border on previous position
                prev_x = last_pos_x + translate_x
                prev_y = last_pos_y + translate_y
                self.draw_border(prev_x, prev_y, self.WHITE)
            else:
                # Replace all blue with white
                self.replace_blue_with_white()
            
            # Draw current position
            self.draw_rectangle(new_pos_x, new_pos_y, self.WHITE)
            self.draw_border(new_pos_x, new_pos_y, self.BLUE)
            
            # Update next flag
            self.next_flag = (translate_x != 0 or translate_y != 0)
        else:
            translate_x = translate_y = 0
        
        # Update total shift
        self.shift_x += translate_x
        self.shift_y += translate_y
        
    
    def get_map(self) -> np.ndarray:
        """Get the current map state."""
        return self.map_camera.copy()
    
    def reset(self) -> None:
        """Reset the map camera to initial state."""
        self.map_camera = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.position_history = []
        self.shift_x = 0
        self.shift_y = 0
        self.next_flag = False
