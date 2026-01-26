# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node built with the v3 Schema API that enables loading musubi-tuner based LoRAs (including Qwen-Image, Z-Image Turbo, Hunyuan Video 1.5, FLUX, and Wan2.1) directly without requiring manual conversion to ComfyUI format. The conversion to diffusers-compatible format happens on-the-fly in memory during node execution.

## Architecture

### Core Components

**nodes.py** - Contains the main node implementation:
- `MusubiTunerLoRALoaderModelOnly`: Implements `io.ComfyNode` base class using ComfyUI v3 Schema API
- `convert_to_diffusers()`: Main conversion function that transforms musubi-tuner LoRA format to diffusers-compatible format
- `QWEN_IMAGE_KEYS`: Predefined list of Qwen-Image state dict keys for format detection
- `MusubiTunerLoRALoaderExtension`: Extension class that implements `ComfyExtension` protocol

**__init__.py** - ComfyUI v3 extension entry point:
- Imports and exports `comfy_entrypoint()` function
- No longer uses legacy `NODE_CLASS_MAPPINGS` dictionary

### Conversion Logic

The `convert_to_diffusers()` function handles multiple musubi-tuner LoRA formats through a three-stage process:

1. **Module Name Mapping**:
   - **Qwen-Image**: Uses `QWEN_IMAGE_KEYS` predefined mappings for explicit key translation
   - **Other formats**: Derives module names from LoRA names by replacing underscores with dots
   - Applies format-specific pattern fixes:
     - **Wan2.1**: `.cross.attn.` → `.cross_attn.`, `.self.attn.` → `.self_attn.`, `k.img` → `k_img`, `v.img` → `v_img`
     - **Z-Image**: `.to.q` → `.to_q`, `.to.k` → `.to_k`, `.to.v` → `.to_v`, `.to.out` → `.to_out`, `.feed.forward` → `.feed_forward`
     - **HunyuanVideo/FLUX**: `double.blocks.` → `double_blocks.`, `single.blocks.` → `single_blocks.`, `img.` → `img_`, `txt.` → `txt_`, `attn.` → `attn_`

2. **Weight Key Transformation**:
   - Converts `lora_down` → `{diffusers_prefix}.{module_name}.lora_A.weight`
   - Converts `lora_up` → `{diffusers_prefix}.{module_name}.lora_B.weight`
   - Extracts dimension info from weight shapes (down: dim = shape[0], up: dim = shape[1])

3. **Alpha Scaling**:
   - Scales weights by `sqrt(alpha / dim)` to maintain proper magnitude
   - Applies scaling to both lora_A and lora_B weights
   - Warns if alpha value is missing for a LoRA layer

### Node Execution Flow

1. **Schema Definition** (`define_schema()`):
   - Defines node inputs: `model` (Model), `lora_name` (Combo from folder_paths), `strength_model` (Float: -100.0 to 100.0, default 1.0)
   - Defines single output: `model` (Model)
   - Sets node metadata: ID, display name, category, description

2. **Execution** (`execute()`):
   - Returns immediately if `strength_model == 0` (optimization shortcut)
   - Loads LoRA file using `comfy.utils.load_torch_file()` with safe loading
   - Converts state dict using `convert_to_diffusers()` with `lora_unet_` prefix
   - Applies LoRA to model using `comfy.sd.load_lora_for_models()`
   - Returns modified model wrapped in `io.NodeOutput`

Note: Conversion happens on every execution; there is no caching mechanism in the current implementation.

## Development

### Testing in ComfyUI

This custom node must be tested within a running ComfyUI instance:

1. Place this directory in `ComfyUI/custom_nodes/`
2. Restart ComfyUI or use the "Reload Custom Nodes" option
3. The node appears as "musubi-tuner LoRA Loader" in the node menu under the LoRA category
4. Test with musubi-tuner format LoRAs (Z-Image Turbo or Hunyuan Video 1.5)

### Dependencies

- `torch` - For tensor operations and weight manipulation
- `tqdm` - Progress bar for conversion loop
- `comfy.sd` - ComfyUI's LoRA loading utilities (`load_lora_for_models`)
- `comfy.utils` - SafeTensors file loading (`load_torch_file`)
- `folder_paths` - ComfyUI's path resolution for LoRA files
- `comfy_api.latest` - ComfyUI v3 Schema API (`ComfyExtension`, `io` module)
- `logging` - Standard Python logging for info/warning messages

### Key Implementation Details

- Node uses ComfyUI v3 Schema API with `io.ComfyNode` base class
- Schema definition is declarative using `io.Schema()` with typed inputs/outputs
- Conversion happens in `convert_to_diffusers()` on every execution (no caching)
- If `strength_model == 0`, the node short-circuits and returns the unmodified model wrapped in `io.NodeOutput`
- State dict is built as a new dictionary (`new_weights_sd = {}`), original is not mutated
- Alpha values are extracted first, then applied during weight conversion
- Logging uses Python's standard `logging` module with INFO level for conversion start and WARNING for missing alphas
- Progress bar via `tqdm` shows processing of each key during conversion
- Extension registration uses async `comfy_entrypoint()` function returning `ComfyExtension` instance
- All conversions preserve tensor device and dtype

## Credits

Conversion logic adapted from: https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/convert_lora.py
