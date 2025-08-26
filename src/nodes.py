import torch
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

class IRCArtConverter:
    CATEGORY = "image/conversion"
    
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required": { 
                "image": ("IMAGE",),
                "block_width": ("INT", {"default": 8, "min": 1, "max": 32}),
                "block_height": ("INT", {"default": 15, "min": 1, "max": 32}),
                "half_block_mode": ("BOOLEAN", {"default": False}),
                "color_method": (["dominant", "average", "median"],)
            } 
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "irc_text")
    FUNCTION = "convert_to_irc_art"
    
    def __init__(self):
        # 99-color IRC palette
        self.IRC_COLORS = [
            (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), 
            (165, 42, 42), (255, 0, 255), (255, 165, 0), (255, 255, 0), (144, 238, 144),
            (0, 255, 255), (173, 216, 230), (173, 216, 255), (255, 192, 203), (128, 128, 128),
            (211, 211, 211), (71, 0, 0), (71, 33, 0), (71, 71, 0), (50, 71, 0),
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
        self.irc_colors_tensor = torch.tensor(self.IRC_COLORS, dtype=torch.float32)
    
    def manhattan_distance(self, detected_color, target_colors):
        """Manhattan distance function matching the color transfer node"""
        return np.sum(np.abs(detected_color - target_colors), axis=1)
    
    def apply_color_transfer(self, image):
        """Apply color transfer using KMeans clustering and Manhattan distance matching"""
        # Convert image to numpy array
        img_array = (image * 255.0).cpu().numpy().astype(np.uint8)
        
        # Reshape for clustering
        img_flat = img_array.reshape((-1, 3))
        
        # KMeans clustering
        clustering_model = KMeans(n_clusters=len(self.IRC_COLORS), n_init="auto", random_state=42)
        clustering_model.fit(img_flat)
        
        detected_colors = clustering_model.cluster_centers_.astype(int)
        
        # Match detected colors to IRC palette using Manhattan distance
        target_colors_array = np.array(self.IRC_COLORS)
        closest_colors = []
        
        for color in detected_colors:
            distances = self.manhattan_distance(color, target_colors_array)
            closest_color = target_colors_array[np.argmin(distances)]
            closest_colors.append(closest_color)
        
        closest_colors = np.array(closest_colors)
        
        # Map each pixel to its closest IRC color
        processed_image = closest_colors[clustering_model.labels_].reshape(img_array.shape)
        
        # Convert back to torch tensor
        return torch.from_numpy(processed_image.astype(np.float32) / 255.0)
    
    def get_irc_color_code(self, color_idx):
        """Convert IRC color array index to mIRC color code (1-99)"""
        return color_idx + 1
    
    def get_block_color(self, block, method):
        """Extract representative color from a block using specified method"""
        if method == "average":
            # Simple average of all pixels in the block
            return tuple(block.mean(dim=(0, 1)).cpu().numpy() * 255)
        
        elif method == "median":
            # Median of all pixels in the block
            block_flat = block.reshape(-1, 3)
            median_vals = torch.median(block_flat, dim=0)[0]
            return tuple(median_vals.cpu().numpy() * 255)
        
        elif method == "dominant":
            # Use k-means to find dominant color
            block_flat = block.reshape(-1, 3).cpu().numpy()
            if len(block_flat) < 2:
                return tuple(block_flat[0] * 255)
            
            # Check for color variance - if too uniform, just return average
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
    
    def convert_to_irc_art(self, image, block_width, block_height, half_block_mode, color_method):
        # Input validation
        batch_size, height, width, channels = image.shape
        
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")
        
        # Get the first (and only) image from batch
        img = image[0]  # Shape: (height, width, 3)
        
        # Step 1: Apply color transfer to map image to IRC palette
        color_transferred_img = self.apply_color_transfer(img)
        
        # Adjust block height for half block mode
        actual_block_height = block_height / 2 if half_block_mode else block_height
        
        # Calculate number of blocks
        blocks_x = width // block_width
        blocks_y = int(height // actual_block_height)
        
        # Create output image and text lines
        output = torch.zeros_like(img)
        text_lines = []
        
        # Process each block
        for y in range(blocks_y):
            line_colors = []  # Store color codes for this line
            
            for x in range(blocks_x):
                # Calculate block boundaries
                start_x = x * block_width
                end_x = min(start_x + block_width, width)
                
                if half_block_mode:
                    start_y = round(y * actual_block_height)
                    end_y = round(min((y + 1) * actual_block_height, height))
                else:
                    start_y = y * block_height
                    end_y = min(start_y + block_height, height)
                
                # Extract block from color-transferred image
                block = color_transferred_img[start_y:end_y, start_x:end_x, :]
                
                if block.numel() == 0:
                    line_colors.append(1)  # Default to black
                    continue
                
                # Get representative color for this block (now already in IRC palette)
                block_color = self.get_block_color(block, color_method)
                
                # Find the exact IRC color index since colors are already mapped
                block_color_array = np.array(block_color)
                target_colors_array = np.array(self.IRC_COLORS)
                distances = self.manhattan_distance(block_color_array, target_colors_array)
                color_idx = np.argmin(distances)
                irc_color = self.IRC_COLORS[color_idx]
                irc_code = self.get_irc_color_code(color_idx)
                
                # Store color code for text output
                line_colors.append(irc_code)
                
                # Fill the entire block with the IRC color
                irc_color_tensor = torch.tensor(irc_color, dtype=torch.float32) / 255.0
                output[start_y:end_y, start_x:end_x, :] = irc_color_tensor
            
            # Build IRC text line with color codes
            # Format: \x03{fg},{bg}{char} where fg=bg for solid blocks
            line_text = ""
            for color_code in line_colors:
                # Use the same color for foreground and background, with space character
                line_text += f"\x03{color_code:02d},{color_code:02d} "
            
            text_lines.append(line_text)
        
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
                "filename": ("STRING", {"default": "irc_art.txt"}),
            } 
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_irc_text"
    OUTPUT_NODE = True
    
    def save_irc_text(self, irc_text, filename):
        """Save IRC text to a file"""
        import os
        from datetime import datetime
        
        # Ensure filename has .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        # Create output directory if it doesn't exist
        output_dir = "output/irc_art"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create full file path with timestamp if file exists
        base_name = filename.replace('.txt', '')
        file_path = os.path.join(output_dir, filename)
        
        # If file exists, add timestamp
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.txt"
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