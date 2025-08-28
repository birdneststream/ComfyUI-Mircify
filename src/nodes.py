"""
IRC Art Converter Node for ComfyUI
Converts images to IRC art with text output support for both IRC and ANSI terminals.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import warnings
import os
import json
from typing import Tuple, List, Optional, Union, Dict, Any

try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

# 99-color IRC palette - matches Go IRCColorPalette exactly
IRC_COLORS = [
    (255, 255, 255), (0, 0, 0), (0, 0, 127), (0, 147, 0), (255, 0, 0),
    (127, 0, 0), (156, 0, 156), (252, 127, 0), (255, 255, 0), (0, 252, 0),
    (0, 147, 147), (0, 255, 255), (0, 0, 252), (255, 0, 255), (127, 127, 127),
    (210, 210, 210), (71, 0, 0), (71, 33, 0), (71, 71, 0), (50, 71, 0),
    (0, 71, 0), (0, 71, 44), (0, 71, 71), (0, 39, 71), (0, 0, 71),
    (46, 0, 71), (71, 0, 71), (71, 0, 42), (116, 0, 0), (116, 58, 0),
    (116, 116, 0), (81, 116, 0), (0, 116, 0), (0, 116, 73), (0, 116, 116),
    (0, 64, 116), (0, 0, 116), (75, 0, 116), (116, 0, 116), (116, 0, 69),
    (181, 0, 0), (181, 99, 0), (181, 181, 0), (125, 181, 0), (0, 181, 0),
    (0, 181, 113), (0, 181, 181), (0, 99, 181), (0, 0, 181), (117, 0, 181),
    (181, 0, 181), (181, 0, 107), (255, 0, 0), (255, 140, 0), (255, 255, 0),
    (178, 255, 0), (0, 255, 0), (0, 255, 160), (0, 255, 255), (0, 140, 255),
    (0, 0, 255), (165, 0, 255), (255, 0, 255), (255, 0, 152), (255, 89, 89),
    (255, 180, 89), (255, 255, 113), (207, 255, 96), (111, 255, 111), (101, 255, 201),
    (109, 255, 255), (89, 180, 255), (89, 89, 255), (196, 89, 255), (255, 102, 255),
    (255, 89, 188), (255, 156, 156), (255, 211, 156), (255, 255, 156), (226, 255, 156),
    (156, 255, 156), (156, 255, 219), (156, 255, 255), (156, 211, 255), (156, 156, 255),
    (220, 156, 255), (255, 156, 255), (255, 148, 211), (0, 0, 0), (19, 19, 19),
    (40, 40, 40), (54, 54, 54), (77, 77, 77), (101, 101, 101), (129, 129, 129),
    (159, 159, 159), (188, 188, 188), (226, 226, 226), (255, 255, 255)
]

# ANSI standard colors (0-15)
ANSI_STANDARD_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
]

# RGB values for 6x6x6 cube mapping
RGB_CUBE_VALUES = [0, 95, 135, 175, 215, 255]

# Distance method weights
DISTANCE_WEIGHTS = {
    'weighted_manhattan': np.array([0.7, 1.0, 0.5]),  # Human visual sensitivity
    'perceptual_weighted': np.array([0.299, 0.587, 0.114])  # Luminance weights
}

# Block processing defaults
DEFAULT_BLOCK_WIDTH = 8
DEFAULT_BLOCK_HEIGHT = 15
HALF_BLOCK_DIVISOR = 2
MIN_COLOR_VARIANCE = 0.01
MAX_KMEANS_CLUSTERS = 3
KMEANS_N_INIT = 10

# Character constants
HALF_BLOCK_CHAR = "▀"
SPACE_CHAR = " "
ANSI_RESET = "\033[0m"


# ============================================================================
# COLOR UTILITIES
# ============================================================================

class ColorUtils:
    """Utility class for color-related operations."""
    
    @staticmethod
    def get_ansi_rgb(color_num: int) -> Tuple[int, int, int]:
        """Get RGB values for ANSI 256 color number."""
        if color_num <= 7:
            return ANSI_STANDARD_COLORS[color_num]
        elif color_num <= 15:
            return ANSI_STANDARD_COLORS[color_num]
        elif color_num <= 231:
            # 6x6x6 RGB cube
            idx = color_num - 16
            r = idx // 36
            g = (idx % 36) // 6
            b = idx % 6
            return (RGB_CUBE_VALUES[r], RGB_CUBE_VALUES[g], RGB_CUBE_VALUES[b])
        else:
            # Grayscale 232-255
            gray = 8 + (color_num - 232) * 10
            return (gray, gray, gray)
    
    @staticmethod
    def find_closest_ansi_color(target_rgb: Tuple[int, int, int]) -> int:
        """Find the closest ANSI 256 color to the target RGB."""
        min_distance = float('inf')
        closest_color = 0
        
        for i in range(256):
            ansi_rgb = ColorUtils.get_ansi_rgb(i)
            # Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(target_rgb, ansi_rgb)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color = i
        
        return closest_color
    
    @staticmethod
    def generate_irc_to_ansi_mapping() -> List[int]:
        """Generate IRC to ANSI color mapping based on RGB distance."""
        return [ColorUtils.find_closest_ansi_color(irc_rgb) for irc_rgb in IRC_COLORS]


class DistanceCalculator:
    """Handles various color distance calculation methods."""
    
    @staticmethod
    def calculate(detected_color: np.ndarray, target_colors: np.ndarray, 
                  method: str) -> np.ndarray:
        """Calculate color distance using the specified method."""
        
        if method == "manhattan":
            return np.sum(np.abs(detected_color - target_colors), axis=1)
        
        elif method == "weighted_manhattan":
            weights = DISTANCE_WEIGHTS['weighted_manhattan']
            diff = np.abs(detected_color - target_colors)
            return np.sum(diff * weights, axis=1)
        
        elif method == "euclidean":
            return np.sqrt(np.sum((detected_color - target_colors) ** 2, axis=1))
        
        elif method == "perceptual_weighted":
            weights = DISTANCE_WEIGHTS['perceptual_weighted']
            diff = (detected_color - target_colors) ** 2
            return np.sqrt(np.sum(diff * weights, axis=1))
        
        # Default fallback to weighted_manhattan
        return DistanceCalculator.calculate(detected_color, target_colors, "weighted_manhattan")


# ============================================================================
# TEXT FORMATTERS
# ============================================================================

class TextFormatter:
    """Base class for text formatting operations."""
    
    @staticmethod
    def count_consecutive(items: List, index: int) -> int:
        """Count consecutive identical entries starting from index."""
        if index >= len(items):
            return 0
            
        count = 1
        current = items[index]
        
        while index + count < len(items) and items[index + count] == current:
            count += 1
        
        return count


class IRCFormatter(TextFormatter):
    """Handles IRC-specific text formatting."""
    
    @staticmethod
    def format_color_code(fg_color: int, bg_color: Optional[int] = None) -> str:
        """Format IRC color code with proper padding."""
        if bg_color is None:
            bg_color = fg_color
        
        fg_fmt = f"{fg_color:02d}" if fg_color >= 10 else str(fg_color)
        bg_fmt = f"{bg_color:02d}" if bg_color >= 10 else str(bg_color)
        return f"\x03{fg_fmt},{bg_fmt}"
    
    @staticmethod
    def format_line(line_colors: List[Union[int, Tuple[int, int]]]) -> str:
        """Format a line of IRC color codes with comprehensive duplicate prevention."""
        if not line_colors:
            return ""
        
        result = []
        i = 0
        last_color_code = None  # Track the last color code to prevent duplicates
        
        while i < len(line_colors):
            color_entry = line_colors[i]
            
            if isinstance(color_entry, tuple):
                # Half-block mode
                top_color, bottom_color = color_entry
                
                if top_color == bottom_color:
                    # Same colors - treat as spaces with background color
                    count = 1
                    while (i + count < len(line_colors) and 
                           isinstance(line_colors[i + count], tuple) and 
                           line_colors[i + count] == color_entry):
                        count += 1
                    
                    color_code = IRCFormatter.format_color_code(top_color)
                    if color_code != last_color_code:
                        result.append(color_code + (SPACE_CHAR * count))
                        last_color_code = color_code
                    else:
                        # Same color code as previous, just add spaces without color code
                        result.append(SPACE_CHAR * count)
                    i += count
                else:
                    # Different colors - check for consecutive identical half-block patterns
                    count = 1
                    while (i + count < len(line_colors) and 
                           isinstance(line_colors[i + count], tuple) and 
                           line_colors[i + count] == color_entry):
                        count += 1
                    
                    color_code = IRCFormatter.format_color_code(top_color, bottom_color)
                    if color_code != last_color_code:
                        result.append(color_code + (HALF_BLOCK_CHAR * count))
                        last_color_code = color_code
                    else:
                        # Same color code as previous, just add characters without color code
                        result.append(HALF_BLOCK_CHAR * count)
                    i += count
            else:
                # Full-block mode - treat as spaces with background color
                count = IRCFormatter.count_consecutive(line_colors, i)
                color_code = IRCFormatter.format_color_code(color_entry)
                if color_code != last_color_code:
                    result.append(color_code + (SPACE_CHAR * count))
                    last_color_code = color_code
                else:
                    # Same color code as previous, just add spaces without color code
                    result.append(SPACE_CHAR * count)
                i += count
        
        return ''.join(result)


class ANSIFormatter(TextFormatter):
    """Handles ANSI terminal text formatting."""
    
    def __init__(self, irc_to_ansi_mapping: List[int]):
        self.mapping = irc_to_ansi_mapping
    
    def get_ansi_color(self, irc_idx: int) -> int:
        """Get ANSI color for IRC color index with bounds checking."""
        if 0 <= irc_idx < len(self.mapping):
            return self.mapping[irc_idx]
        return self.mapping[0]  # Default to first color
    
    def format_background(self, ansi_color: int, count: int = 1) -> str:
        """Format ANSI background color with spaces."""
        return f"\033[48;5;{ansi_color}m" + (SPACE_CHAR * count)
    
    def format_half_block(self, top_ansi: int, bottom_ansi: int) -> str:
        """Format ANSI half-block with foreground and background colors."""
        return f"\033[38;5;{top_ansi};48;5;{bottom_ansi}m{HALF_BLOCK_CHAR}"
    
    def format_line(self, line_colors: List[Union[int, Tuple[int, int]]]) -> str:
        """Format a line of ANSI color codes with comprehensive duplicate prevention."""
        if not line_colors:
            return ""
        
        result = []
        i = 0
        last_ansi_code = None  # Track the last ANSI code to prevent duplicates
        
        while i < len(line_colors):
            color_entry = line_colors[i]
            
            if isinstance(color_entry, tuple):
                # Half-block mode
                top_color, bottom_color = color_entry
                
                if top_color == bottom_color:
                    # Same colors - treat as background spaces
                    count = 1
                    while (i + count < len(line_colors) and 
                           isinstance(line_colors[i + count], tuple) and 
                           line_colors[i + count] == color_entry):
                        count += 1
                    
                    ansi_color = self.get_ansi_color(top_color)
                    ansi_code = f"\033[48;5;{ansi_color}m"
                    if ansi_code != last_ansi_code:
                        result.append(ansi_code + (SPACE_CHAR * count))
                        last_ansi_code = ansi_code
                    else:
                        # Same ANSI code as previous, just add spaces without code
                        result.append(SPACE_CHAR * count)
                    i += count
                else:
                    # Different colors - check for consecutive identical patterns
                    count = 1
                    while (i + count < len(line_colors) and 
                           isinstance(line_colors[i + count], tuple) and 
                           line_colors[i + count] == color_entry):
                        count += 1
                    
                    top_ansi = self.get_ansi_color(top_color)
                    bottom_ansi = self.get_ansi_color(bottom_color)
                    ansi_code = f"\033[38;5;{top_ansi};48;5;{bottom_ansi}m"
                    if ansi_code != last_ansi_code:
                        result.append(ansi_code + (HALF_BLOCK_CHAR * count))
                        last_ansi_code = ansi_code
                    else:
                        # Same ANSI code as previous, just add characters without code
                        result.append(HALF_BLOCK_CHAR * count)
                    i += count
            else:
                # Full-block mode - treat as background spaces
                count = ANSIFormatter.count_consecutive(line_colors, i)
                ansi_color = self.get_ansi_color(color_entry)
                ansi_code = f"\033[48;5;{ansi_color}m"
                if ansi_code != last_ansi_code:
                    result.append(ansi_code + (SPACE_CHAR * count))
                    last_ansi_code = ansi_code
                else:
                    # Same ANSI code as previous, just add spaces without code
                    result.append(SPACE_CHAR * count)
                i += count
        
        result.append(ANSI_RESET)
        return ''.join(result)


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class BlockProcessor:
    """Handles block-based image processing operations."""
    
    @staticmethod
    def get_dominant_color(block: torch.Tensor) -> Tuple[int, int, int]:
        """Extract dominant color from a block using k-means clustering."""
        block_flat = block.reshape(-1, 3).cpu().numpy()
        
        # Handle edge cases
        if len(block_flat) < 2:
            return tuple((block_flat[0] * 255).astype(int))
        
        # Check for uniform color
        color_std = np.std(block_flat, axis=0)
        if np.max(color_std) < MIN_COLOR_VARIANCE:
            return tuple((np.mean(block_flat, axis=0) * 255).astype(int))
        
        # Determine cluster count
        unique_colors = np.unique(block_flat, axis=0)
        n_clusters = min(MAX_KMEANS_CLUSTERS, len(unique_colors))
        
        if n_clusters == 1:
            return tuple((unique_colors[0] * 255).astype(int))
        
        # Apply k-means clustering
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=KMEANS_N_INIT)
                kmeans.fit(block_flat)
            
            # Find dominant cluster
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            return tuple((dominant_color * 255).astype(int))
        except Exception:
            # Fallback to average
            return tuple((block.mean(dim=(0, 1)).cpu().numpy() * 255).astype(int))
    
    @staticmethod
    def extract_block(image: torch.Tensor, x: int, y: int, 
                     block_width: int, actual_block_height: float,
                     img_width: int, img_height: int) -> torch.Tensor:
        """Extract a block from the image with proper boundaries."""
        start_x = x * block_width
        end_x = min(start_x + block_width, img_width)
        start_y = round(y * actual_block_height)
        end_y = round(min((y + 1) * actual_block_height, img_height))
        
        return image[start_y:end_y, start_x:end_x, :]


class ColorTransfer:
    """Handles color transfer operations to map images to IRC palette."""
    
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calc = distance_calculator
    
    def apply(self, image: torch.Tensor, use_16_colors: bool = False, 
              distance_method: str = "weighted_manhattan") -> torch.Tensor:
        """Apply color transfer using KMeans clustering and distance matching."""
        # Convert to numpy
        img_array = (image * 255.0).cpu().numpy().astype(np.uint8)
        img_flat = img_array.reshape((-1, 3))
        
        # Select palette
        target_colors = IRC_COLORS[:16] if use_16_colors else IRC_COLORS
        
        # Apply KMeans
        clustering_model = KMeans(n_clusters=len(target_colors), n_init="auto", random_state=42)
        clustering_model.fit(img_flat)
        
        detected_colors = clustering_model.cluster_centers_.astype(int)
        
        # Match to IRC palette
        target_colors_array = np.array(target_colors)
        closest_colors = []
        
        for color in detected_colors:
            distances = self.distance_calc.calculate(color, target_colors_array, distance_method)
            closest_colors.append(target_colors_array[np.argmin(distances)])
        
        closest_colors = np.array(closest_colors)
        
        # Map pixels
        processed_image = closest_colors[clustering_model.labels_].reshape(img_array.shape)
        
        return torch.from_numpy(processed_image.astype(np.float32) / 255.0)


# ============================================================================
# PNG EXIF UTILITIES
# ============================================================================

class PNGExifHandler:
    """Handles PNG metadata embedding operations."""
    
    @staticmethod
    def create_metadata_json(irc_text: str = None, ansi_text: str = None, 
                           block_width: int = None, block_height: int = None,
                           half_block_mode: bool = None, color_method: str = None) -> str:
        """Create JSON metadata string with IRC art data."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "generator": "ComfyUI-Mircify",
            "version": "1.0"
        }
        
        # Add text data if provided
        if irc_text is not None:
            metadata["irc"] = irc_text
        if ansi_text is not None:
            metadata["ansi"] = ansi_text
        
        # Add optional parameters if provided
        if block_width is not None:
            metadata["block_width"] = block_width
        if block_height is not None:
            metadata["block_height"] = block_height
        if half_block_mode is not None:
            metadata["half_block_mode"] = half_block_mode
        if color_method is not None:
            metadata["color_method"] = color_method
            
        return json.dumps(metadata, ensure_ascii=False)
    
    @staticmethod
    def embed_metadata(image: Image.Image, metadata_json: str) -> Image.Image:
        """Embed metadata into PNG UserComment EXIF field."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for EXIF metadata embedding")
        
        # Get existing EXIF data or create new
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        
        if hasattr(image, '_getexif') and image._getexif() is not None:
            # Copy existing EXIF data
            existing_exif = image._getexif()
            for tag_id, value in existing_exif.items():
                if tag_id in TAGS:
                    tag_name = TAGS[tag_id]
                    if tag_name in ["UserComment", "ImageDescription"]:
                        continue  # We'll override these
                    exif_dict["0th"][tag_id] = value
        
        # Add our metadata to UserComment (0x9286)
        # UserComment format: charset (8 bytes) + actual comment
        # Using ASCII charset identifier
        charset_identifier = b"ASCII\x00\x00\x00"
        user_comment = charset_identifier + metadata_json.encode('utf-8', errors='ignore')
        exif_dict["Exif"][ExifTags.Base.UserComment.value] = user_comment
        
        return image
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor to PIL Image."""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy and scale to 0-255
        numpy_image = tensor.cpu().numpy()
        if numpy_image.max() <= 1.0:
            numpy_image = (numpy_image * 255).astype(np.uint8)
        else:
            numpy_image = numpy_image.astype(np.uint8)
            
        # Convert to PIL
        return Image.fromarray(numpy_image, 'RGB')
    
    @staticmethod
    def save_png_with_metadata(tensor: torch.Tensor, filepath: str, metadata_json: str) -> None:
        """Save tensor as PNG with embedded metadata."""
        pil_image = PNGExifHandler.tensor_to_pil(tensor)
        
        # Create PNG info for metadata
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        
        # Add metadata as PNG text chunks (more reliable than EXIF for PNG)
        png_info.add_text("UserComment", metadata_json)
        png_info.add_text("Software", "ComfyUI-Mircify")
        png_info.add_text("Description", "IRC Art with embedded text data")
        
        # Save with metadata
        pil_image.save(filepath, format='PNG', pnginfo=png_info, optimize=True)


# ============================================================================
# MAIN NODE CLASSES
# ============================================================================

class IRCArtConverter:
    """Main node for converting images to IRC art."""
    
    CATEGORY = "image/conversion"
    
    def __init__(self):
        self.distance_calc = DistanceCalculator()
        self.color_transfer = ColorTransfer(self.distance_calc)
        self.block_processor = BlockProcessor()
        self._irc_to_ansi_mapping = None
        self._ansi_formatter = None
        self._irc_formatter = IRCFormatter()
    
    @property
    def ansi_formatter(self) -> ANSIFormatter:
        """Lazy initialization of ANSI formatter."""
        if self._ansi_formatter is None:
            if self._irc_to_ansi_mapping is None:
                self._irc_to_ansi_mapping = ColorUtils.generate_irc_to_ansi_mapping()
            self._ansi_formatter = ANSIFormatter(self._irc_to_ansi_mapping)
        return self._ansi_formatter
    
    def _is_output_connected(self, prompt: Optional[Dict], node_id: Optional[str], output_index: int) -> bool:
        """Check if a specific output is connected to another node."""
        if not prompt or not node_id:
            return True  # Default to generating output if we can't determine connection status
        
        try:
            # Check all nodes in the prompt to see if any use this node's output
            for node_data in prompt.values():
                if isinstance(node_data, dict) and "inputs" in node_data:
                    inputs = node_data["inputs"]
                    for input_value in inputs.values():
                        # ComfyUI represents connections as [node_id, output_index]
                        if (isinstance(input_value, list) and 
                            len(input_value) == 2 and 
                            input_value[0] == node_id and 
                            input_value[1] == output_index):
                            return True
            return False
        except Exception:
            # If there's any error in parsing, default to generating the output
            return True
    
    @classmethod    
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return { 
            "required": { 
                "image": ("IMAGE",),
                "block_width": ("INT", {"default": DEFAULT_BLOCK_WIDTH, "min": 1, "max": 32}),
                "block_height": ("INT", {"default": DEFAULT_BLOCK_HEIGHT, "min": 1, "max": 32}),
                "half_block_mode": ("BOOLEAN", {"default": False}),
                "use_16_colors": ("BOOLEAN", {"default": False}),
                "distance_method": (["weighted_manhattan", "manhattan", "euclidean", "perceptual_weighted"], 
                                   {"default": "perceptual_weighted"}),
            } 
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("processed_image", "irc_text", "ansi_text")
    FUNCTION = "convert_to_irc_art"
    
    def find_irc_color_index(self, color: Tuple[int, int, int], use_16_colors: bool, 
                            distance_method: str) -> int:
        """Find the IRC color index for a given color."""
        # Select palette
        palette = IRC_COLORS[:16] if use_16_colors else IRC_COLORS
        
        # Try exact match first
        color_rounded = tuple(np.round(color).astype(int))
        try:
            return palette.index(color_rounded)
        except ValueError:
            # Fallback to distance matching
            target_colors_array = np.array(palette)
            distances = self.distance_calc.calculate(np.array(color), target_colors_array, distance_method)
            return int(np.argmin(distances))
    
    def process_block(self, image: torch.Tensor, x: int, y: int, params: Dict[str, Any]) -> Tuple[int, int, int]:
        """Process a single block and return its color index and boundaries."""
        block = self.block_processor.extract_block(
            image, x, y, params['block_width'], params['actual_block_height'],
            params['width'], params['height']
        )
        
        # Calculate boundaries for output
        start_y = round(y * params['actual_block_height'])
        end_y = round(min((y + 1) * params['actual_block_height'], params['height']))
        
        if block.numel() == 0:
            return 0, start_y, end_y
        
        block_color = self.block_processor.get_dominant_color(block)
        color_idx = self.find_irc_color_index(
            block_color, params['use_16_colors'], params['distance_method']
        )
        
        return color_idx, start_y, end_y
    
    def apply_color_to_output(self, output: torch.Tensor, color_idx: int,
                            start_y: int, end_y: int, start_x: int, end_x: int) -> None:
        """Apply IRC color to output image region."""
        irc_color = torch.tensor(IRC_COLORS[color_idx], dtype=torch.float32) / 255.0
        output[start_y:end_y, start_x:end_x, :] = irc_color
    
    def process_row(self, image: torch.Tensor, y: int, output: torch.Tensor,
                   params: Dict[str, Any], half_block: bool = False) -> List[Union[int, Tuple[int, int]]]:
        """Process a row of blocks."""
        line_colors = []
        blocks_x = params['blocks_x']
        
        for x in range(blocks_x):
            # Process first block
            color_idx1, start_y1, end_y1 = self.process_block(image, x, y, params)
            
            # Apply to output
            x_start = x * params['block_width']
            x_end = min((x + 1) * params['block_width'], params['width'])
            self.apply_color_to_output(output, color_idx1, start_y1, end_y1, x_start, x_end)
            
            if half_block:
                # Process second half-block if exists
                if y + 1 < params['blocks_y']:
                    color_idx2, start_y2, end_y2 = self.process_block(image, x, y + 1, params)
                    self.apply_color_to_output(output, color_idx2, start_y2, end_y2, x_start, x_end)
                    line_colors.append((color_idx1, color_idx2))
                else:
                    line_colors.append((color_idx1, color_idx1))
            else:
                line_colors.append(color_idx1)
        
        return line_colors
    
    def convert_to_irc_art(self, image: torch.Tensor, block_width: int, block_height: int,
                          half_block_mode: bool, use_16_colors: bool, distance_method: str, 
                          prompt=None, extra_pnginfo=None, my_unique_id=None) -> Tuple[torch.Tensor, str, str]:
        """Main conversion function."""
        # Check which outputs are connected
        irc_connected = self._is_output_connected(prompt, my_unique_id, 1)  # irc_text is output index 1
        ansi_connected = self._is_output_connected(prompt, my_unique_id, 2)  # ansi_text is output index 2
        
        # Validate input
        batch_size, height, width, channels = image.shape
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        img = image[0]
        
        # Apply color transfer
        color_transferred_img = self.color_transfer.apply(img, use_16_colors, distance_method)
        
        # Calculate block dimensions
        blocks_x = width // block_width
        
        if half_block_mode:
            blocks_y = (height * 2) // block_height
            actual_block_height = block_height / HALF_BLOCK_DIVISOR
        else:
            blocks_y = height // block_height
            actual_block_height = float(block_height)
        
        # Prepare parameters
        params = {
            'block_width': block_width,
            'block_height': block_height,
            'actual_block_height': actual_block_height,
            'width': width,
            'height': height,
            'blocks_x': blocks_x,
            'blocks_y': blocks_y,
            'use_16_colors': use_16_colors,
            'distance_method': distance_method
        }
        
        # Process image
        output = torch.zeros_like(img)
        irc_lines = []
        ansi_lines = []
        
        if half_block_mode:
            # Process in pairs for half-block mode
            for y in range(0, blocks_y, 2):
                line_colors = self.process_row(color_transferred_img, y, output, params, half_block=True)
                if irc_connected:
                    irc_lines.append(self._irc_formatter.format_line(line_colors))
                if ansi_connected:
                    ansi_lines.append(self.ansi_formatter.format_line(line_colors))
        else:
            # Process full blocks
            for y in range(blocks_y):
                line_colors = self.process_row(color_transferred_img, y, output, params, half_block=False)
                if irc_connected:
                    irc_lines.append(self._irc_formatter.format_line(line_colors))
                if ansi_connected:
                    ansi_lines.append(self.ansi_formatter.format_line(line_colors))
        
        # Format output
        irc_text = "\n".join(irc_lines) if irc_connected else ""
        ansi_text = "\n".join(ansi_lines) if ansi_connected else ""
        
        return (output.unsqueeze(0), irc_text, ansi_text)


class IRCTextSaver:
    """Node to save IRC formatted text to a file."""
    
    CATEGORY = "text/output"
    OUTPUT_DIR = "output/irc_art"
    DEFAULT_FILENAME = "irc_art_{TIMESTAMP}.txt"
    
    @classmethod    
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return { 
            "required": { 
                "irc_text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": cls.DEFAULT_FILENAME}),
            } 
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_irc_text"
    OUTPUT_NODE = True
    
    def save_irc_text(self, irc_text: str, filename: str) -> Dict[str, Any]:
        """Save IRC text to a file."""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process filename
        if "{TIMESTAMP}" in filename:
            filename = filename.replace("{TIMESTAMP}", timestamp)
        
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        # Create output directory
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Save file
        file_path = os.path.join(self.OUTPUT_DIR, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(irc_text)
            
            abs_path = os.path.abspath(file_path)
            success_msg = f"IRC text saved to: {abs_path}"
            print(success_msg)
            
            return {"ui": {"text": [success_msg]}, "result": (abs_path,)}
            
        except Exception as e:
            error_msg = f"Error saving file: {str(e)}"
            print(error_msg)
            return {"ui": {"text": [error_msg]}, "result": (error_msg,)}


class IRCPNGExporter:
    """Node to export images as PNG with embedded IRC/ANSI metadata."""
    
    CATEGORY = "image/output"
    OUTPUT_DIR = "output/irc_art"
    DEFAULT_FILENAME = "irc_art_{TIMESTAMP}.png"
    
    @classmethod    
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return { 
            "required": { 
                "image": ("IMAGE",),
                "preview": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "irc_text": ("STRING", {"forceInput": True}),
                "ansi_text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": cls.DEFAULT_FILENAME}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "export_png_with_metadata"
    OUTPUT_NODE = True
    
    def export_png_with_metadata(self, image: torch.Tensor, preview: bool, 
                                irc_text: str = None, ansi_text: str = None, 
                                filename: str = None) -> Dict[str, Any]:
        """Export PNG with embedded IRC/ANSI metadata."""
        
        # Check PIL availability
        if not PIL_AVAILABLE:
            error_msg = "PIL/Pillow is required for PNG export with metadata. Please install with: pip install Pillow"
            return {"ui": {"text": [error_msg]}}
        
        # Create metadata JSON
        metadata_json = PNGExifHandler.create_metadata_json(irc_text, ansi_text)
        metadata_dict = json.loads(metadata_json)
        
        # Check if we have any text data to embed
        has_text_data = irc_text is not None or ansi_text is not None
        text_info = []
        if irc_text is not None:
            text_info.append("IRC text")
        if ansi_text is not None:
            text_info.append("ANSI text")
        
        if preview:
            # Preview mode - create a temporary image with metadata and display it
            import tempfile
            import uuid
            
            if has_text_data:
                metadata_preview = f"Preview mode - Image with embedded metadata ({', '.join(text_info)})"
            else:
                metadata_preview = f"Preview mode - Image with basic metadata (no text data connected)"
            
            # Create temporary file with metadata for preview using ComfyUI's approach
            try:
                import folder_paths
                import random
                
                # Use ComfyUI's temp directory structure
                temp_dir = folder_paths.get_temp_directory()
                
                # Generate filename like ComfyUI's PreviewImage node
                temp_filename = f"preview_{random.randint(0, 2**32):08x}.png"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                # Convert tensor to PIL image using ComfyUI's method
                i = 255.0 * image[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Add metadata using PNG text chunks (using standard visible fields)
                from PIL.PngImagePlugin import PngInfo
                png_info = PngInfo()
                
                # Use standard PNG text chunks that are more likely to be visible
                png_info.add_text("Description", f"IRC Art with embedded data")
                png_info.add_text("Comment", metadata_json)  # Main data in Comment field
                png_info.add_text("Software", "ComfyUI-Mircify")
                png_info.add_text("Author", "ComfyUI IRC Art Converter")
                
                # Save with metadata
                img.save(temp_path, pnginfo=png_info, compress_level=4)
                
                # Return with proper ComfyUI image display format
                return {
                    "ui": {
                        "images": [{
                            "filename": temp_filename,
                            "subfolder": "",
                            "type": "temp"
                        }],
                        "text": [metadata_preview]
                    }
                }
                
            except Exception as e:
                error_msg = f"Error creating preview: {str(e)}"
                return {"ui": {"text": [error_msg]}}
        
        else:
            # Save mode - save PNG with metadata using ComfyUI's approach
            if not filename:
                filename = self.DEFAULT_FILENAME
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Process filename
            if "{TIMESTAMP}" in filename:
                filename = filename.replace("{TIMESTAMP}", timestamp)
            
            if not filename.lower().endswith('.png'):
                filename += '.png'
            
            # Create output directory
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            
            # Save file with metadata
            file_path = os.path.join(self.OUTPUT_DIR, filename)
            
            try:
                # Convert tensor to PIL image using ComfyUI's method
                i = 255.0 * image[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                # Add metadata using PNG text chunks (using standard visible fields)
                from PIL.PngImagePlugin import PngInfo
                png_info = PngInfo()
                
                # Use standard PNG text chunks that are more likely to be visible
                png_info.add_text("Description", f"IRC Art with embedded data")
                png_info.add_text("Comment", metadata_json)  # Main data in Comment field
                png_info.add_text("Software", "ComfyUI-Mircify")
                png_info.add_text("Author", "ComfyUI IRC Art Converter")
                
                # Save with metadata
                img.save(file_path, pnginfo=png_info, compress_level=4)
                
                abs_path = os.path.abspath(file_path)
                
                if has_text_data:
                    success_msg = f"PNG with metadata saved to: {abs_path}\nEmbedded data: {', '.join(text_info)}"
                else:
                    success_msg = f"PNG with basic metadata saved to: {abs_path}\n(No text data was connected)"
                
                return {"ui": {"text": [success_msg]}}
                
            except Exception as e:
                error_msg = f"Error saving PNG with metadata: {str(e)}"
                return {"ui": {"text": [error_msg]}}


# ============================================================================
# NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "IRC Art Converter": IRCArtConverter,
    "IRC Text Saver": IRCTextSaver,
    "IRC PNG Exporter": IRCPNGExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IRC Art Converter": "IRC Art Converter", 
    "IRC Text Saver": "IRC Text Saver",
    "IRC PNG Exporter": "IRC PNG Exporter",
}