import torch
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from datetime import datetime

class IRCArtConverter:
    CATEGORY = "image/conversion"
    
    # 99-color IRC palette (class constant) - matches Go IRCColorPalette exactly
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
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "image": ("IMAGE",),
                "block_width": ("INT", {"default": 8, "min": 1, "max": 32}),
                "block_height": ("INT", {"default": 15, "min": 1, "max": 32}),
                "half_block_mode": ("BOOLEAN", {"default": False}),
                "use_16_colors": ("BOOLEAN", {"default": False}),
                "distance_method": (["weighted_manhattan", "manhattan", "euclidean", "perceptual_weighted"], {"default": "weighted_manhattan"}),
            } 
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "irc_text")
    FUNCTION = "convert_to_irc_art"
    
    def calculate_color_distance(self, detected_color, target_colors, method):
        """Calculate color distance using various methods."""
        if method == "manhattan":
            # Standard Manhattan distance (L1 norm)
            return np.sum(np.abs(detected_color - target_colors), axis=1)
        
        elif method == "weighted_manhattan":
            # Weighted Manhattan distance for better perceptual matching
            # Weights based on human visual sensitivity: [R, G, B]
            weights = np.array([0.7, 1.0, 0.5])
            diff = np.abs(detected_color - target_colors)
            return np.sum(diff * weights, axis=1)
        
        elif method == "euclidean":
            # Standard Euclidean distance (L2 norm)
            return np.sqrt(np.sum((detected_color - target_colors) ** 2, axis=1))
        
        elif method == "perceptual_weighted":
            # Perceptual weighted Euclidean distance
            # Based on CIE recommendations for RGB weights
            weights = np.array([0.299, 0.587, 0.114])  # Luminance weights
            diff = (detected_color - target_colors) ** 2
            return np.sqrt(np.sum(diff * weights, axis=1))
        
        else:
            # Fallback to weighted manhattan
            weights = np.array([0.7, 1.0, 0.5])
            diff = np.abs(detected_color - target_colors)
            return np.sum(diff * weights, axis=1)
    
    def apply_color_transfer(self, image, use_16_colors=False, distance_method="weighted_manhattan"):
        """Apply color transfer using KMeans clustering and configurable distance matching."""
        # Convert image to numpy array
        img_array = (image * 255.0).cpu().numpy().astype(np.uint8)
        img_flat = img_array.reshape((-1, 3))
        
        # Select color palette based on mode
        target_colors = self.IRC_COLORS[:16] if use_16_colors else self.IRC_COLORS
        
        # KMeans clustering
        clustering_model = KMeans(n_clusters=len(target_colors), n_init="auto", random_state=42)
        clustering_model.fit(img_flat)
        
        detected_colors = clustering_model.cluster_centers_.astype(int)
        
        # Match detected colors to IRC palette using Manhattan distance
        target_colors_array = np.array(target_colors)
        closest_colors = []
        
        for color in detected_colors:
            distances = self.calculate_color_distance(color, target_colors_array, distance_method)
            closest_colors.append(target_colors_array[np.argmin(distances)])
        
        closest_colors = np.array(closest_colors)
        
        # Map each pixel to its closest IRC color
        processed_image = closest_colors[clustering_model.labels_].reshape(img_array.shape)
        
        # Convert back to torch tensor
        return torch.from_numpy(processed_image.astype(np.float32) / 255.0)
    
    def get_block_color(self, block):
        """Extract dominant color from a block using k-means clustering."""
        # Use k-means to find dominant color
        block_flat = block.reshape(-1, 3).cpu().numpy()
        if len(block_flat) < 2:
            return tuple(block_flat[0] * 255)
        
        # Check for color variance
        color_std = np.std(block_flat, axis=0)
        if np.max(color_std) < 0.01:  # Very uniform color
            return tuple(np.mean(block_flat, axis=0) * 255)
        
        # Find unique colors to determine appropriate cluster count
        unique_colors = np.unique(block_flat, axis=0)
        n_clusters = min(3, len(unique_colors))
        
        if n_clusters == 1:
            return tuple(unique_colors[0] * 255)
        
        # Use k-means with appropriate cluster count
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(block_flat)
            
            # Find the cluster with most points
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            return tuple(dominant_color * 255)
        except:
            # Fallback to average if k-means fails
            return tuple(block.mean(dim=(0, 1)).cpu().numpy() * 255)
    
    def find_irc_color_index(self, block_color, use_16_colors, distance_method="weighted_manhattan"):
        """Find the IRC color index for a given block color."""
        # Get representative color for this block
        block_color_rounded = tuple(np.round(block_color).astype(int))
        
        # Find the exact IRC color index using exact matching first
        # Try exact match first (since colors should already be in IRC palette)
        try:
            if use_16_colors:
                color_idx = self.IRC_COLORS[:16].index(block_color_rounded)
            else:
                color_idx = self.IRC_COLORS.index(block_color_rounded)
        except ValueError:
            # Fallback to distance matching if exact match fails
            block_color_array = np.array(block_color)
            if use_16_colors:
                target_colors_array = np.array(self.IRC_COLORS[:16])
            else:
                target_colors_array = np.array(self.IRC_COLORS)
            
            distances = self.calculate_color_distance(block_color_array, target_colors_array, distance_method)
            color_idx = np.argmin(distances)
        
        return color_idx
    
    def format_irc_line(self, line_colors):
        """Format a line of IRC color codes with compression."""
        if not line_colors:
            return ""
        
        line_text = ""
        i = 0
        while i < len(line_colors):
            color_code = line_colors[i]
            count = 1
            
            # Count consecutive identical colors
            while i + count < len(line_colors) and line_colors[i + count] == color_code:
                count += 1
            
            # Format color code
            if color_code <= 9:
                color_format = f"\x03{color_code},{color_code}"
            else:
                color_format = f"\x03{color_code:02d},{color_code:02d}"
            
            line_text += color_format + (" " * count)
            i += count
        
        return line_text
    
    def process_block(self, color_transferred_img, x, y, block_width, block_height, 
                     actual_block_height, width, height, use_16_colors, distance_method):
        """Process a single block and return its color index."""
        # Calculate block boundaries
        start_x = x * block_width
        end_x = min(start_x + block_width, width)
        start_y = round(y * actual_block_height)
        end_y = round(min((y + 1) * actual_block_height, height))
        
        # Extract block
        block = color_transferred_img[start_y:end_y, start_x:end_x, :]
        
        # Get IRC color index
        if block.numel() == 0:
            return 0, start_y, end_y
        else:
            block_color = self.get_block_color(block)
            return self.find_irc_color_index(block_color, use_16_colors, distance_method), start_y, end_y
    
    def convert_to_irc_art(self, image, block_width, block_height, half_block_mode, use_16_colors, distance_method):
        # Input validation
        batch_size, height, width, channels = image.shape
        
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        # Get the first (and only) image from batch
        img = image[0]  # Shape: (height, width, 3)
        
        # Apply color transfer to map image to IRC palette
        color_transferred_img = self.apply_color_transfer(img, use_16_colors, distance_method)
        
        # Calculate number of blocks
        blocks_x = width // block_width
        
        if half_block_mode:
            blocks_y = (height * 2) // block_height
            actual_block_height = block_height / 2
        else:
            blocks_y = height // block_height
            actual_block_height = block_height
        
        # Create output image and text lines
        output = torch.zeros_like(img)
        text_lines = []
        
        # Process blocks
        if half_block_mode:
            # Half-block mode: process in pairs
            for y in range(0, blocks_y, 2):
                line_colors = []
                
                for x in range(blocks_x):
                    # Process first half-block
                    color_idx, start_y, end_y = self.process_block(
                        color_transferred_img, x, y, block_width, block_height,
                        actual_block_height, width, height, use_16_colors, distance_method
                    )
                    
                    # Apply color to image and store for text
                    irc_color = self.IRC_COLORS[color_idx]
                    irc_color_tensor = torch.tensor(irc_color, dtype=torch.float32) / 255.0
                    output[start_y:end_y, x * block_width:min((x + 1) * block_width, width), :] = irc_color_tensor
                    line_colors.append(color_idx + 1)  # IRC codes are 1-based
                    
                    # Process second half-block if it exists
                    if y + 1 < blocks_y:
                        color_idx2, start_y2, end_y2 = self.process_block(
                            color_transferred_img, x, y + 1, block_width, block_height,
                            actual_block_height, width, height, use_16_colors, distance_method
                        )
                        
                        # Apply color to second block
                        irc_color2 = self.IRC_COLORS[color_idx2]
                        irc_color_tensor2 = torch.tensor(irc_color2, dtype=torch.float32) / 255.0
                        output[start_y2:end_y2, x * block_width:min((x + 1) * block_width, width), :] = irc_color_tensor2
                
                text_lines.append(self.format_irc_line(line_colors))
        else:
            # Full block mode
            for y in range(blocks_y):
                line_colors = []
                
                for x in range(blocks_x):
                    color_idx, start_y, end_y = self.process_block(
                        color_transferred_img, x, y, block_width, block_height,
                        actual_block_height, width, height, use_16_colors, distance_method
                    )
                    
                    # Apply color to image and store for text
                    irc_color = self.IRC_COLORS[color_idx]
                    irc_color_tensor = torch.tensor(irc_color, dtype=torch.float32) / 255.0
                    output[start_y:end_y, x * block_width:min((x + 1) * block_width, width), :] = irc_color_tensor
                    line_colors.append(color_idx + 1)  # IRC codes are 1-based
                
                text_lines.append(self.format_irc_line(line_colors))
        
        # Join all lines with newlines
        irc_text = "\n".join(text_lines)
        
        # Return as batch of 1
        return (output.unsqueeze(0), irc_text)


class IRCTextSaver:
    """Node to save IRC formatted text to a file"""
    CATEGORY = "text/output"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "irc_text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"default": "irc_art_{TIMESTAMP}.txt"}),
            } 
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_irc_text"
    OUTPUT_NODE = True
    
    def save_irc_text(self, irc_text, filename):
        """Save IRC text to a file"""
        import os
        
        # Generate current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Replace {TIMESTAMP} placeholder in filename if present
        if "{TIMESTAMP}" in filename:
            filename = filename.replace("{TIMESTAMP}", timestamp)
        
        # Ensure filename has .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        # Create output directory if it doesn't exist
        output_dir = "output/irc_art"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full file path
        file_path = os.path.join(output_dir, filename)
        
        # Write the IRC text to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(irc_text)
            
            abs_path = os.path.abspath(file_path)
            success_msg = f"✅ IRC text saved to: {abs_path}"
            print(success_msg)
            
            return {"ui": {"text": [success_msg]}, "result": (abs_path,)}
            
        except Exception as e:
            error_msg = f"❌ Error saving file: {str(e)}"
            print(error_msg)
            return {"ui": {"text": [error_msg]}, "result": (error_msg,)}

NODE_CLASS_MAPPINGS = {
    "IRC Art Converter": IRCArtConverter,
    "IRC Text Saver": IRCTextSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IRC Art Converter": "IRC Art Converter", 
    "IRC Text Saver": "IRC Text Saver",
}