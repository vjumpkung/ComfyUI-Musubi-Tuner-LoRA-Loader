# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node that enables loading musubi-tuner based LoRAs (such as Z-Image Turbo and Hunyuan Video 1.5) directly without requiring manual conversion to ComfyUI format. The conversion happens on-the-fly in memory during node execution.

## Architecture

### Core Components

**nodes.py** - Contains the main node implementation:
- `MusubiTunerLoRALoaderModelOnly`: Extends ComfyUI's built-in `LoraLoaderModelOnly` class
- Performs runtime conversion of musubi-tuner LoRA format to ComfyUI-compatible format
- Implements two conversion strategies:
  1. **Direct key renaming** - Handles layer naming differences between formats
  2. **QKV concatenation** - Merges separate Q/K/V LoRA weights into unified QKV layers

**__init__.py** - Standard ComfyUI custom node entry point that exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`

### Conversion Logic

The node handles two types of musubi-tuner LoRAs:

1. **Z-Image Turbo** (uses `layers` namespace):
   - Renames: `attention_to_out_0` → `attention_out`, `attention_norm_k` → `attention_k_norm`, `attention_norm_q` → `attention_q_norm`
   - Merges `to_q`, `to_k`, `to_v` → `qkv` layers

2. **Hunyuan Video 1.5** (uses `double_blocks` namespace):
   - Renames: `img_mlp_fc1` → `img_mlp_0`, `img_mlp_fc2` → `img_mlp_2`, `img_mod_linear` → `img_mod_lin`, and corresponding `txt_*` variants
   - Merges `_q`, `_k`, `_v` → `_qkv` layers

**QKV Merging Strategy**: The conversion concatenates separate Q/K/V LoRA down-weights and creates a sparse up-weight matrix. The alpha value is multiplied by 3 to account for the 3x larger rank (see kohya-ss/sd-scripts#2204).

### State Management

The node caches the converted LoRA state dict in `self.loaded_lora` as a tuple of `(lora_path, state_dict)` to avoid redundant conversions when the same LoRA is used multiple times.

## Development

### Testing in ComfyUI

This custom node must be tested within a running ComfyUI instance:

1. Place this directory in `ComfyUI/custom_nodes/`
2. Restart ComfyUI or use the "Reload Custom Nodes" option
3. The node appears as "musubi-tuner LoRA Loader" in the node menu under the LoRA category
4. Test with musubi-tuner format LoRAs (Z-Image Turbo or Hunyuan Video 1.5)

### Dependencies

- `torch` - For tensor operations during QKV merging
- `tqdm` - Progress bar for QKV conversion loop
- `comfy.sd` - ComfyUI's LoRA loading utilities
- `comfy.utils` - SafeTensors file loading
- `folder_paths` - ComfyUI's path resolution for model files
- `nodes.LoraLoaderModelOnly` - Parent class from ComfyUI core

### Key Implementation Details

- Conversion happens in `load_musubi_tuner_lora()` on first load, then cached
- If `strength_model == 0`, the node short-circuits and returns the unmodified model
- The state dict is modified in-place using `state_dict.pop()` and `state_dict[new_key] = ...`
- Logging uses Python's standard `logging` module with INFO level by default
- All conversions preserve tensor device and dtype

## Credits

Conversion logic adapted from: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora_zimage.py
