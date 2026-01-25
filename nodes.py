import logging

import torch
from tqdm import tqdm

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        """
        - This node is loading and coverting musubi-tuner based LoRA like Z-Image Turbo and Hunyuan Video 1.5
          without create new file and ready to use in ComfyUI.
          by converting before load into ComfyUI.
        - Credits : https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/networks/lora_zimage.py
        """
        if strength_model == 0:
            return io.NodeOutput(
                model,
            )

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None

        if lora is None:
            logger.info(
                "Start Converting Musubi Tuner LoRA to ComfyUI Compatible Format..."
            )

            state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)

            keys = list(state_dict.keys())
            count = 0

            # for musubi-tuner z-image lora
            blocks_mappings = [
                ("attention_to_out_0", "attention_out"),
                ("attention_norm_k", "attention_k_norm"),
                ("attention_norm_q", "attention_q_norm"),
            ]

            # for musubi-tuner hunyuan video 1.5 lora
            double_blocks_mappings = [
                ("img_mlp_fc1", "img_mlp_0"),
                ("img_mlp_fc2", "img_mlp_2"),
                ("img_mod_linear", "img_mod_lin"),
                ("txt_mlp_fc1", "txt_mlp_0"),
                ("txt_mlp_fc2", "txt_mlp_2"),
                ("txt_mod_linear", "txt_mod_lin"),
            ]

            for key in keys:
                new_k = key

                if "layers" in key:
                    mappings = blocks_mappings
                elif "double_blocks" in key:
                    mappings = double_blocks_mappings
                else:
                    continue

                # Apply mappings based on conversion direction
                for src_key, dst_key in mappings:
                    new_k = new_k.replace(src_key, dst_key)

                if new_k != key:
                    state_dict[new_k] = state_dict.pop(key)
                    count += 1
                    # print(f"Renamed {key} to {new_k}")

            # sd-scripts to ComfyUI: concat QKV
            qkv_count = 0
            keys = list(state_dict.keys())
            for key in tqdm(keys, desc="Processing QKV layers"):
                # for z-image turbo
                if "attention" in key and (
                    "to_q" in key or "to_k" in key or "to_v" in key
                ):
                    if (
                        "to_q" not in key or "lora_up" not in key
                    ):  # ensure we process only once per QKV set
                        continue

                    lora_name = key.split(".", 1)[0]  # get LoRA base name
                    split_dims = [
                        state_dict[key].size(0)
                    ] * 3  # assume equal split for Q, K, V

                    lora_name_prefix = lora_name.replace("to_q", "")
                    down_weights = []  # (rank, in_dim) * 3
                    up_weights = []  # (split dim, rank) * 3
                    for weight_index in range(len(split_dims)):
                        if weight_index == 0:
                            suffix = "to_q"
                        elif weight_index == 1:
                            suffix = "to_k"
                        else:
                            suffix = "to_v"
                        down_weights.append(
                            state_dict.pop(
                                f"{lora_name_prefix}{suffix}.lora_down.weight"
                            )
                        )
                        up_weights.append(
                            state_dict.pop(f"{lora_name_prefix}{suffix}.lora_up.weight")
                        )

                    alpha = state_dict.pop(f"{lora_name}.alpha")
                    state_dict.pop(f"{lora_name_prefix}to_k.alpha")
                    state_dict.pop(f"{lora_name_prefix}to_v.alpha")

                    # merge down weight
                    down_weight = torch.cat(
                        down_weights, dim=0
                    )  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

                    # merge up weight (sum of split_dim, rank*3), dense to sparse
                    rank = up_weights[0].size(1)
                    up_weight = torch.zeros(
                        (sum(split_dims), down_weight.size(0)),
                        device=down_weight.device,
                        dtype=down_weight.dtype,
                    )
                    weight_index = 0
                    for i in range(len(split_dims)):
                        up_weight[
                            weight_index : weight_index + split_dims[i],
                            i * rank : (i + 1) * rank,
                        ] = up_weights[i]
                        weight_index += split_dims[i]

                    new_lora_name = lora_name_prefix + "qkv"
                    state_dict[f"{new_lora_name}.lora_down.weight"] = down_weight
                    state_dict[f"{new_lora_name}.lora_up.weight"] = up_weight

                    # adjust alpha because rank is 3x larger. See https://github.com/kohya-ss/sd-scripts/issues/2204
                    state_dict[f"{new_lora_name}.alpha"] = alpha * 3
                    qkv_count += 1

                # for hunyuan video 1.5
                elif ("img_attn" in key or "txt_attn" in key) and (
                    "_q" in key or "_k" in key or "_v" in key
                ):
                    if (
                        "_q" not in key or "lora_up" not in key
                    ):  # ensure we process only once per QKV set
                        continue

                    lora_name = key.split(".", 1)[0]  # get LoRA base name
                    split_dims = [
                        state_dict[key].size(0)
                    ] * 3  # assume equal split for Q, K, V

                    lora_name_prefix = lora_name.replace("_q", "")
                    down_weights = []  # (rank, in_dim) * 3
                    up_weights = []  # (split dim, rank) * 3
                    for weight_index in range(len(split_dims)):
                        if weight_index == 0:
                            suffix = "_q"
                        elif weight_index == 1:
                            suffix = "_k"
                        else:
                            suffix = "_v"
                        down_weights.append(
                            state_dict.pop(
                                f"{lora_name_prefix}{suffix}.lora_down.weight"
                            )
                        )
                        up_weights.append(
                            state_dict.pop(f"{lora_name_prefix}{suffix}.lora_up.weight")
                        )

                    alpha = state_dict.pop(f"{lora_name}.alpha")
                    state_dict.pop(f"{lora_name_prefix}_k.alpha")
                    state_dict.pop(f"{lora_name_prefix}_v.alpha")

                    # merge down weight
                    down_weight = torch.cat(
                        down_weights, dim=0
                    )  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

                    # merge up weight (sum of split_dim, rank*3), dense to sparse
                    rank = up_weights[0].size(1)
                    up_weight = torch.zeros(
                        (sum(split_dims), down_weight.size(0)),
                        device=down_weight.device,
                        dtype=down_weight.dtype,
                    )
                    weight_index = 0
                    for i in range(len(split_dims)):
                        up_weight[
                            weight_index : weight_index + split_dims[i],
                            i * rank : (i + 1) * rank,
                        ] = up_weights[i]
                        weight_index += split_dims[i]

                    new_lora_name = lora_name_prefix + "_qkv"
                    state_dict[f"{new_lora_name}.lora_down.weight"] = down_weight
                    state_dict[f"{new_lora_name}.lora_up.weight"] = up_weight

                    # adjust alpha because rank is 3x larger. See https://github.com/kohya-ss/sd-scripts/issues/2204
                    state_dict[f"{new_lora_name}.alpha"] = alpha * 3

                    qkv_count += 1

            if count == 0 and qkv_count == 0:
                logger.warning(
                    "This LoRA does not need to converted to ComfyUI format."
                )
            else:
                logger.info(f"Direct key renames applied: {count}")
                logger.info(f"QKV LoRA layers processed: {qkv_count}")
                logger.info("Convert LoRA to ComfyUI format successfully.")

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
