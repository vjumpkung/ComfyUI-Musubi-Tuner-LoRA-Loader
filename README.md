# ComfyUI Musubi-Tuner LoRA Loader

A ComfyUI custom node that enables direct loading of musubi-tuner based LoRAs (Z-Image Turbo, Hunyuan Video 1.5) without manual format conversion. The conversion happens automatically in-memory during node execution.

## Features

- **On-the-fly Conversion**: Automatically converts musubi-tuner LoRA format to ComfyUI-compatible format
- **No File Creation**: All conversion happens in-memory, no temporary files needed
- **Caching**: Converted LoRAs are cached to avoid redundant processing
- **Dual Format Support**:
  - Z-Image Turbo LoRAs (uses `layers` namespace)
  - Hunyuan Video 1.5 LoRAs (uses `double_blocks` namespace)

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ComfyUI-Musubi-Tuner-LoRA-Loader.git
   ```

3. Restart ComfyUI or reload custom nodes

## Usage

1. Find the node in ComfyUI under the name **"musubi-tuner LoRA Loader"**
2. Connect a MODEL input
3. Select your musubi-tuner format LoRA from the dropdown
4. Adjust the strength_model parameter (default: 1.0, range: -100.0 to 100.0)
5. Connect the MODEL output to your workflow

## How It Works

The node performs two main conversion operations:

### 1. Direct Key Renaming
Maps musubi-tuner layer names to ComfyUI equivalents:
- **Z-Image**: `attention_to_out_0` → `attention_out`, etc.
- **Hunyuan Video**: `img_mlp_fc1` → `img_mlp_0`, etc.

### 2. QKV Layer Merging
Combines separate Q/K/V LoRA weights into unified QKV layers:
- Concatenates down-weights from Q, K, V layers
- Creates sparse up-weight matrix
- Adjusts alpha values (×3) to account for increased rank

## Technical Details

- **Parent Class**: Extends `nodes.LoraLoaderModelOnly` from ComfyUI core
- **State Caching**: Converted LoRAs are stored as `(path, state_dict)` tuples
- **Optimization**: Returns original model immediately if `strength_model == 0`
- **Dependencies**: `torch`, `tqdm`, ComfyUI core modules

## Credits

Conversion logic adapted from [musubi-tuner by kohya-ss](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora_zimage.py)

## License

This project is provided as-is for use with ComfyUI.
