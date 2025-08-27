# ComfyUI-Mircify

A ComfyUI custom node for converting images into IRC art by processing them in pixel blocks and generating both image and text outputs with optimal color selection from a 99-color IRC palette.

This node is used with [aibird](https://github.com/birdneststream/aibird) to convert images to IRC art.

![Workflow Example](workflows/screenshot.png)

## Workflow

An example workflow is provided in `workflows/mircify_example_workflow.json`. At the moment this example is using qwen image lightning, however any model that is decent at making pixel ansi style art can suffice.

## Features

- **Universal Image Support**: Works with any image dimensions
- **Color Transfer Integration**: Uses KMeans clustering with Manhattan distance for accurate color mapping
- **Block Processing**: Configurable block sizes (default 8x15 pixels)
- **Half Block Mode**: Option for 8x7.5 pixel blocks for finer resolution
- **Multiple Color Methods**: Choose between dominant, average, or median color extraction
- **99-Color IRC Palette**: Maps colors to extended IRC color palette
- **16-Color Compatibility Mode**: Option to use traditional 16-color IRC palette
- **Dual Output**: Returns both processed image and mIRC-formatted text
- **Optimized Text Output**: Compressed color codes and space-efficient formatting
- **Text File Export**: Save IRC codes directly to text files with timestamp support

## Installation

1. Clone or download this repository to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/birdneststream/ComfyUI-Mircify
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-Mircify
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Nodes

### IRC Art Converter
**Category**: `image/conversion`

Converts images into IRC art with both visual and text output.

**Inputs:**
- **image**: Input image (any dimensions supported)
- **block_width**: Width of each block in pixels (default: 8)
- **block_height**: Height of each block in pixels (default: 15)
- **half_block_mode**: Enable half-height blocks (8x7.5 instead of 8x15)
- **use_16_colors**: Use only first 16 IRC colors for compatibility (default: False)
- **color_method**: Method for color selection (dominant/average/median)

**Outputs:**
- **processed_image**: Visual representation of the IRC art conversion
- **irc_text**: mIRC-formatted text with color codes ready for IRC clients

### IRC Text Saver
**Category**: `text/output`

Saves IRC-formatted text to files for easy copying.

**Inputs:**
- **irc_text**: IRC text from the converter (STRING input)
- **filename**: Output filename with {TIMESTAMP} placeholder support (default: "irc_art.txt")

**Outputs:**
- **file_path**: Absolute path to the saved file

## Usage

1. **Basic Workflow:**
   - Load an image
   - Connect it to "IRC Art Converter"
   - Connect the `irc_text` output to "IRC Text Saver"
   - Run the workflow

2. **Color Methods:**
   - **dominant**: Uses k-means clustering to find the most prominent color in each block
   - **average**: Calculates the mean RGB values for the block
   - **median**: Uses the median RGB values for better noise resistance

3. **Block Modes:**
   - **Full blocks (8x15)**: Standard IRC character block size
   - **Half blocks (8x7.5)**: Provides finer vertical resolution

4. **Color Modes:**
   - **99-color mode**: Full extended IRC palette for maximum color accuracy
   - **16-color mode**: Traditional IRC colors for better compatibility

5. **Text Output:**
   - Uses optimized mIRC color text output
   - Ready to paste directly into IRC clients
   - Support for {TIMESTAMP} placeholder in filenames

## Color Processing

The node uses a two-stage approach:

1. **Color Transfer Phase**: Applies KMeans clustering with Manhattan distance to map the entire image to the 99-color IRC palette
2. **Block Processing Phase**: Divides the color-corrected image into blocks and generates the final IRC art output

## IRC Color Palette

The node uses an extended 99-color IRC palette built in.

## Requirements

- ComfyUI
- Python 3.8+
- torch
- numpy
- scikit-learn
- pillow

## Credits

- Color transfer logic adapted from [ComfyUI-Color_Transfer](https://github.com/45uee/ComfyUI-Color_Transfer) by 45uee
- Uses KMeans clustering and Manhattan distance for optimal color mapping

## License

MIT License