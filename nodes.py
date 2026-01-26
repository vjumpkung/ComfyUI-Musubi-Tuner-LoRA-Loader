import logging

import torch
from tqdm import tqdm

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
- This node is loading and coverting musubi-tuner based LoRA like Z-Image Turbo and Hunyuan Video 1.5
    without create new file and ready to use in ComfyUI.
    by converting before load into ComfyUI.
- Credits : https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/convert_lora.py
"""

# keys of Qwen-Image state dict
QWEN_IMAGE_KEYS = [
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "txt_norm",
    "img_in",
    "txt_in",
    "transformer_blocks.*.img_mod.1",
    "transformer_blocks.*.attn.norm_q",
    "transformer_blocks.*.attn.norm_k",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.add_k_proj",
    "transformer_blocks.*.attn.add_v_proj",
    "transformer_blocks.*.attn.add_q_proj",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.attn.to_add_out",
    "transformer_blocks.*.attn.norm_added_q",
    "transformer_blocks.*.attn.norm_added_k",
    "transformer_blocks.*.img_mlp.net.0.proj",
    "transformer_blocks.*.img_mlp.net.2",
    "transformer_blocks.*.txt_mod.1",
    "transformer_blocks.*.txt_mlp.net.0.proj",
    "transformer_blocks.*.txt_mlp.net.2",
    "norm_out.linear",
    "proj_out",
]


def convert_to_diffusers(prefix, diffusers_prefix, weights_sd):
    # convert from default LoRA to diffusers
    if diffusers_prefix is None:
        diffusers_prefix = "diffusion_model"

    # make reverse map from LoRA name to base model module name
    lora_name_to_module_name = {}
    for key in QWEN_IMAGE_KEYS:
        if "*" not in key:
            lora_name = prefix + key.replace(".", "_")
            lora_name_to_module_name[lora_name] = key
        else:
            lora_name = prefix + key.replace(".", "_")
            for i in range(100):  # assume at most 100 transformer blocks
                lora_name_to_module_name[lora_name.replace("*", str(i))] = key.replace(
                    "*", str(i)
                )

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}
    for key, weight in tqdm(weights_sd.items(), desc="Processing QKV layers"):
        if key.startswith(prefix):
            if "alpha" in key:
                continue

            lora_name = key.split(".", 1)[0]  # before first dot

            if lora_name in lora_name_to_module_name:
                module_name = lora_name_to_module_name[lora_name]
            else:
                module_name = lora_name[len(prefix) :]  # remove "lora_unet_"
                module_name = module_name.replace("_", ".")  # replace "_" with "."
                if ".cross.attn." in module_name or ".self.attn." in module_name:
                    # Wan2.1 lora name to module name: ugly but works
                    module_name = module_name.replace(
                        "cross.attn", "cross_attn"
                    )  # fix cross attn
                    module_name = module_name.replace(
                        "self.attn", "self_attn"
                    )  # fix self attn
                    module_name = module_name.replace("k.img", "k_img")  # fix k img
                    module_name = module_name.replace("v.img", "v_img")  # fix v img
                elif ".attention.to." in module_name or ".feed.forward." in module_name:
                    # Z-Image lora name to module name: ugly but works
                    module_name = module_name.replace("to.q", "to_q")  # fix to q
                    module_name = module_name.replace("to.k", "to_k")  # fix to k
                    module_name = module_name.replace("to.v", "to_v")  # fix to v
                    module_name = module_name.replace("to.out", "to_out")  # fix to out
                    module_name = module_name.replace(
                        "feed.forward", "feed_forward"
                    )  # fix feed forward
                elif "double.blocks." in module_name or "single.blocks." in module_name:
                    # HunyuanVideo and FLUX lora name to module name: ugly but works
                    module_name = module_name.replace(
                        "double.blocks.", "double_blocks."
                    )  # fix double blocks
                    module_name = module_name.replace(
                        "single.blocks.", "single_blocks."
                    )  # fix single blocks
                    module_name = module_name.replace("img.", "img_")  # fix img
                    module_name = module_name.replace("txt.", "txt_")  # fix txt
                    module_name = module_name.replace("attn.", "attn_")  # fix attn

            if "lora_down" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusers_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.warning(f"unexpected key: {key} in default LoRA format")
                continue

            # scale weight by alpha
            if lora_name in lora_alphas:
                # we scale both down and up, so scale is sqrt
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.warning(f"missing alpha for {lora_name}")

            new_weights_sd[new_key] = weight

    return new_weights_sd


class MusubiTunerLoRALoaderModelOnly(io.ComfyNode):
    @classmethod
    def define_schema(self) -> io.Schema:
        return io.Schema(
            node_id="MusubiTunerLoRALoaderModelOnly",
            display_name="musubi-tuner LoRA Loader",
            category="musubi-tuner-lora-loader",
            description="This node is loading and coverting musubi-tuner based LoRA like Z-Image Turbo and Hunyuan Video 1.5 without create new file and ready to use in ComfyUI.",
            inputs=[
                io.Model.Input(
                    "model",
                ),
                io.Combo.Input(
                    "lora_name", options=folder_paths.get_filename_list("loras")
                ),
                io.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(self, model, lora_name, strength_model):
        if strength_model == 0:
            return io.NodeOutput(
                model,
            )

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        logger.info(
            "Start Converting Musubi Tuner LoRA to ComfyUI Compatible Format..."
        )
        
        prefix = "lora_unet_"
        state_dict = convert_to_diffusers(prefix, None, state_dict)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, None, state_dict, strength_model, None
        )

        return io.NodeOutput(
            model_lora,
        )


class MusubiTunerLoRALoaderExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MusubiTunerLoRALoaderModelOnly]


async def comfy_entrypoint() -> ComfyExtension:
    return MusubiTunerLoRALoaderExtension()
