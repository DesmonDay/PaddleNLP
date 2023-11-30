# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
import json
import os

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from tqdm.auto import tqdm

from paddlenlp.trainer.utils.helper import distributed_file, distributed_isfile
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _load_state_dict_into_model,
    get_parameter_dtype,
    load_state_dict,
    unwrap_model,
)
from paddlenlp.transformers.utils import (
    device_guard,
    dtype_byte_size,
    get_checkpoint_shard_files,
    is_safetensors_available,
)
from paddlenlp.utils.distributed import distributed_gather
from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from paddlenlp.utils.log import logger

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.numpy import save_file as safe_save_file


PADDLE_OPTIMIZER_NAME = "optimizer.pdopt"
PADDLE_OPTIMIZER_INDEX_NAME = "optimizer.pdopt.index.json"
SAFE_OPTIMIZER_NAME = "optimizer.safetensors"
SAFE_OPTIMIZER_INDEX_NAME = "optimizer.safetensors.index.json"
PADDLE_MASTER_WEIGHTS_NAME = "master_weights.pdparams"
PADDLE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.pdparams.index.json"
SAFE_MASTER_WEIGHTS_NAME = "master_weights.safetensors"
SAFE_MASTER_WEIGHTS_INDEX_NAME = "master_weights.safetensors.index.json"


__all__ = [
    "load_unified_checkpoint",
    "save_unified_checkpoint",
    "load_unified_optimizer",
    "save_unified_optimizer",
]


def save_unified_checkpoint(args, model, output_dir, safe_serialization=False):
    """save unified checkpoint

    Args:
        args (TrainingArguments): Training Arguments
        model (PretrainedModel): model to save
        output_dir (str): save dir
        safe_serialization (bool, optional): use safetensors. Defaults to False.

    Raises:
        ValueError: if model is not an instance of `PretrainedModel` and the model cannot be saved
    """
    if isinstance(model, PretrainedModel):
        model_to_save = model
    elif isinstance(unwrap_model(model), PretrainedModel):
        model_to_save = unwrap_model(model)
    else:
        raise ValueError("Unified checkpoint only supports PretrainedModel")

    config_to_save = None
    state_dict, config_to_save, shard_file, sharded_index = unified_checkpoint_into_shards(
        args, model_to_save, safe_serialization=safe_serialization
    )

    save_directory = output_dir
    os.makedirs(save_directory, exist_ok=True)
    if safe_serialization:
        for k in list(state_dict.keys()):
            if isinstance(state_dict[k], paddle.Tensor):
                state_dict[k] = state_dict.pop(k).cpu().numpy()
        safe_save_file(state_dict, os.path.join(save_directory, shard_file), metadata={"format": "np"})
    else:
        paddle.save(state_dict, os.path.join(save_directory, shard_file))

    # Attach architecture to the config
    config_to_save.architectures = [model_to_save.__class__.__name__]
    # Save the config
    if args.should_save:
        config_to_save.save_pretrained(save_directory)

    if sharded_index is not None:
        if not safe_serialization:
            path = os.path.join(output_dir, PADDLE_WEIGHTS_INDEX_NAME)
        else:
            path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)

        with open(path, "w") as f:
            json.dump(sharded_index, f, indent=4)


def load_unified_checkpoint(args, model, resume_from_checkpoint: str, safe_serialization=False) -> None:
    """Load potential model checkpoint

    Args:
        model (PretrainedModel): Your model to load
        resume_from_checkpoint (str): path of the checkpoint to load

    Returns:
        None
    """

    resume_inplace = check_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization)
    if not resume_inplace:
        logger.info("Begin to dynamically load unified checkpoint!")
        dynamic_load_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization)
        return

    if args.dataset_rank == 0:
        index_filename = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME

        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename),
        )

        loaded_keys = sharded_metadata["all_checkpoint_keys"]

        model_state_dict = model.state_dict()
        expected_keys = set(list(model_state_dict.keys()))
        missing_keys = expected_keys - set(loaded_keys)

        if len(missing_keys) > 0:
            raise ValueError(f"missing_keys: {missing_keys}")

        def _remove_unused_keys(
            state_dict,
            model_state_dict,
        ):
            unused_keys = set(state_dict.keys()) - set(model_state_dict.keys())
            for unused_key in unused_keys:
                del state_dict[unused_key]
            return unused_keys

        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        error_msgs = []

        if len(resolved_archive_file) > 1:
            resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

        for shard_file in resolved_archive_file:
            # TODO: check if  no expected_keys in shard_file, then don't load it
            if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                continue

            pre_tensor_parallel_split = False
            if shard_file.endswith(".safetensors") and model.config.tensor_parallel_degree > 1:
                pre_tensor_parallel_split = True
                assert loaded_keys is not None, "loaded_keys is not None."
                tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys, ignore_error=True)
            # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
            state_dict = load_state_dict(shard_file, tp_actions if pre_tensor_parallel_split else None, expected_keys)

            if not pre_tensor_parallel_split:
                # Since we load all keys but we only need one of pipeline stages
                _ = _remove_unused_keys(state_dict, model_state_dict)

            if model.config.tensor_parallel_degree > 1 and not pre_tensor_parallel_split:
                logger.info("Converting state_dict to Tensor Parallel Format")
                # ignore error for multi shard, since only parts of data
                state_dict = model.convert_tensor_parallel(
                    None, model.config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
                )
            error_msgs += _load_state_dict_into_model(model, state_dict, "")

            # force memory release
            del state_dict
            gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def unified_checkpoint_into_shards(
    args,
    model_to_save,
    safe_serialization=False,
):
    """Get state_dict and config to save

    Args:
        model_to_save (nn.Layer): model to, save
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.

    Returns:
        tuple: state_dict, config, shard_file: file name, sharded_index: map for weight to file name.
    """
    assert hasattr(model_to_save, "config")

    state_dict = model_to_save.state_dict()

    all_filter_keys = filter_params(model_to_save, state_dict)

    dtype = get_parameter_dtype(model_to_save)
    model_to_save.config.dtype = str(dtype).split(".")[1]
    config_to_save = copy.deepcopy(model_to_save.config)

    if config_to_save.tensor_parallel_degree > 1:
        tp_actions = model_to_save.get_tensor_parallel_convert_actions(
            model_to_save.config, state_dict.keys(), is_split=False, ignore_error=True
        )
        state_dict = merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys)

    if config_to_save.tensor_parallel_degree > 1:
        # do we need to change?
        config_to_save.tensor_parallel_degree = 1

    # build index json file
    index_weight_file = {}
    total_size = 0
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME

    shard_file = get_sharded_file_name(args, weights_name)
    for key, weight in state_dict.items():
        index_weight_file[key] = shard_file
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_file_list, total_size_list = gather_sharded_object(index_weight_file, total_size)
    sharded_index = get_sharded_index(
        index_file_list,
        total_size_list,
    )

    return state_dict, config_to_save, shard_file, sharded_index


def save_unified_optimizer(args, model, optimizer, output_dir, safe_serialization=False):
    """save unified optimizer

    Args:
        args (TrainingArguments): Training Arguments
        optimizer (Optimizer): optimizer to save
        output_dir (str): Save directory.
        safe_serialization (bool, optional): Whether to use safetensors. Defaults to False.

    """
    # Split into naive optimizer params and master weights.
    results = unified_optimizer_into_shards(args, model, optimizer, safe_serialization=safe_serialization)
    master_weight_state_dict = None
    if len(results) == 1:
        optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
    else:
        optim_state_dict, shard_optim_file, sharded_optim_index = results[0]
        master_weight_state_dict, shard_master_weight_file, sharded_master_weight_index = results[1]

    save_directory = output_dir
    os.makedirs(save_directory, exist_ok=True)

    if safe_serialization:
        for k in list(optim_state_dict.keys()):
            if isinstance(optim_state_dict[k], paddle.Tensor):
                optim_state_dict[k] = optim_state_dict.pop(k).cpu().numpy()
        safe_save_file(optim_state_dict, os.path.join(save_directory, shard_optim_file), metadata={"format": "np"})
        if master_weight_state_dict is not None:
            for k in list(master_weight_state_dict.keys()):
                if isinstance(master_weight_state_dict[k], paddle.Tensor):
                    master_weight_state_dict[k] = master_weight_state_dict.pop(k).cpu().numpy()
            safe_save_file(
                master_weight_state_dict,
                os.path.join(save_directory, shard_master_weight_file),
                metadata={"format": "np"},
            )
    else:
        paddle.save(optim_state_dict, os.path.join(save_directory, shard_optim_file))
        if master_weight_state_dict is not None:
            paddle.save(master_weight_state_dict, os.path.join(save_directory, shard_master_weight_file))

    if sharded_optim_index is not None:
        if not safe_serialization:
            path = os.path.join(output_dir, PADDLE_OPTIMIZER_INDEX_NAME)
            master_path = os.path.join(output_dir, PADDLE_MASTER_WEIGHTS_INDEX_NAME)
        else:
            path = os.path.join(output_dir, SAFE_OPTIMIZER_INDEX_NAME)
            master_path = os.path.join(output_dir, SAFE_MASTER_WEIGHTS_INDEX_NAME)

        with open(path, "w") as f:
            json.dump(sharded_optim_index, f, indent=4)

        if master_weight_state_dict is not None:
            with open(master_path, "w") as f:
                json.dump(sharded_master_weight_index, f, indent=4)


def load_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    """Load potential model checkpoint

    Args:
        model (PretrainedModel): Your model to load
        resume_from_checkpoint (str): path of the checkpoint to load

    Returns:
        None
    """
    resume_inplace = check_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization)
    if not resume_inplace:
        logger.info("Begin to dynamically load unified optimizer!")
        returned_optim_state_dict = dynamic_load_unified_optimizer(
            args, model, optimizer, resume_from_checkpoint, safe_serialization
        )
        return returned_optim_state_dict

    if args.data_parallel_rank == 0:
        # init and get optimizer LR_Scheduler
        returned_optim_state_dict = nested_copy(optimizer.state_dict())

        if not safe_serialization:
            index_filename, index_filename_master_weights = (
                PADDLE_OPTIMIZER_INDEX_NAME,
                PADDLE_MASTER_WEIGHTS_INDEX_NAME,
            )
        else:
            index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME

        resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename),
        )
        has_master_weights = True if sharded_metadata["master_weights"] else False

        model_state_dict = model.state_dict()
        model_keys = list(model_state_dict.keys())
        struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings

        expected_keys = get_expected_keys(sharded_metadata, model, optimizer)

        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]
        if len(resolved_archive_file) > 1:
            resolved_archive_file = tqdm(resolved_archive_file, desc="Loading optimizer shards")

        if has_master_weights:
            returned_optim_state_dict["master_weights"] = {}

            resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
                optimizer_path=resume_from_checkpoint,
                index_filename=os.path.join(resume_from_checkpoint, index_filename_master_weights),
            )

            expected_keys_mw = get_expected_keys(sharded_metadata_mw, model, optimizer)
            if not isinstance(resolved_archive_file_mw, list):
                resolved_archive_file_mw = [resolved_archive_file_mw]
            if len(resolved_archive_file_mw) > 1:
                resolved_archive_file_mw = tqdm(resolved_archive_file_mw, desc="Loading master weights shards")

        def load_resolved_archive_file(
            resolved_archive_file, sharded_metadata, expected_keys, is_master_weights=False
        ):
            returned_state_dict = {}
            # load optimizer
            for shard_file in resolved_archive_file:
                # TODO: check if no expected_keys in shard_file, then don't load it
                if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                    continue

                if shard_file.endswith(".safetensors"):
                    # assert model_keys is not None, "model_keys is None." TODO: correct the assert
                    if model.config.tensor_parallel_degree > 1:
                        tp_actions = model.get_tensor_parallel_convert_actions(
                            model.config, model_keys, ignore_error=True
                        )
                        if not is_master_weights:
                            tp_actions = mapping_optimizer_tp_actions(tp_actions, expected_keys)

                        # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                        state_dict = load_state_dict(shard_file, tp_actions, expected_keys)
                    else:
                        # for pipeline model, we don't need to use tp_actions
                        state_dict = load_state_dict(shard_file, None, expected_keys)

                returned_state_dict.update(state_dict)
                # force memory release
                del state_dict
                gc.collect()
            return returned_state_dict

        state_dict_optim = load_resolved_archive_file(resolved_archive_file, sharded_metadata, expected_keys)
        if has_master_weights:
            state_dict_master_weight = load_resolved_archive_file(
                resolved_archive_file_mw, sharded_metadata_mw, expected_keys_mw, is_master_weights=True
            )

        # rename optimizer param
        for key in list(state_dict_optim.keys()):
            key_name = key.split("/")
            static_name = struct2static_name_mappings[key_name[0]]
            if has_master_weights:
                key_name = "_".join([static_name, "fp32_master_0", key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
            returned_optim_state_dict[key_name] = state_dict_optim[key]
            returned_optim_state_dict[key_name].name = key_name

        if has_master_weights:
            for key in list(state_dict_master_weight.keys()):
                static_name = struct2static_name_mappings[key]
                returned_optim_state_dict["master_weights"][static_name] = state_dict_master_weight[key]
                returned_optim_state_dict["master_weights"][static_name].name = "_".join(
                    [static_name, "fp32_master_0"]
                )

        returned_optim_state_dict = nested_copy_place(
            returned_optim_state_dict, place=paddle.framework._current_expected_place()
        )

        return returned_optim_state_dict
    return None


def unified_optimizer_into_shards(
    args,
    model,
    optimizer,
    safe_serialization=False,
):
    """Get optimizer state dict and master weight state dict.

    Args:
        optimizer (Optimizer): optimizer to save.
        safe_serialization (bool, optional): safe serialization using safetensors. Defaults to False.
    """
    optim_state_dict = nested_copy(optimizer.state_dict())
    master_weights = None
    if "master_weights" in optim_state_dict.keys():
        master_weights = optim_state_dict["master_weights"]
        optim_state_dict.pop("master_weights")
    if "LR_Scheduler" in optim_state_dict.keys():
        optim_state_dict.pop("LR_Scheduler")

    # get optimizer param mappings
    static2struct_name_mappings = {}
    for k, v in model.state_dict().items():
        static2struct_name_mappings[v.name] = k

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)

    # filter optimizer param
    if master_weights is not None:
        filter_master_keys = filter_params(model, master_weights, is_optimizer=True)
    filter_optim_keys = filter_params(model, optim_state_dict, is_optimizer=True)

    tp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    tp_size = tp_group.nranks

    if tp_size > 1:
        # get tp_actions
        model_keys = []
        for key in optim_state_dict.keys():
            base_model_key = key.split("/")[0]
            if base_model_key not in model_keys:
                model_keys.append(base_model_key)
        tp_actions = model.get_tensor_parallel_convert_actions(
            model.config, model_keys, is_split=False, ignore_error=True
        )
        optim_state_dict = merge_tensor_parallel_for_optimizer(
            optim_state_dict,
            tp_actions,
            filter_optim_keys,
        )
        if master_weights is not None:
            master_weights = merge_tensor_parallel_for_optimizer(
                master_weights,
                tp_actions,
                filter_master_keys,
            )

    # build index json file
    index_optimizer_file, index_master_weight_file = {}, {}
    total_optim_size, total_master_weight_size = 0, 0
    optimizer_name = SAFE_OPTIMIZER_NAME if safe_serialization else PADDLE_OPTIMIZER_NAME
    master_weights_name = SAFE_MASTER_WEIGHTS_NAME if safe_serialization else PADDLE_MASTER_WEIGHTS_NAME
    shard_optimizer_file = get_sharded_file_name(args, optimizer_name, is_optimizer=True)
    shard_master_weight_file = get_sharded_file_name(args, master_weights_name, is_optimizer=True)

    for key, weight in optim_state_dict.items():
        index_optimizer_file[key] = shard_optimizer_file
        total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    if master_weights is not None:
        for key, weight in master_weights.items():
            index_master_weight_file[key] = shard_master_weight_file
            total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)

    index_optimizer_filelist, total_optim_size_list = gather_sharded_object(
        index_optimizer_file, total_optim_size, is_optimizer=True
    )
    sharded_optim_index = get_sharded_index(index_optimizer_filelist, total_optim_size_list)
    if master_weights is not None:
        index_master_weight_filelist, total_master_weight_size_list = gather_sharded_object(
            index_master_weight_file, total_master_weight_size, is_optimizer=True
        )
        sharded_master_weight_index = get_sharded_index(index_master_weight_filelist, total_master_weight_size_list)

    if sharded_optim_index is not None:
        if master_weights is not None:
            sharded_optim_index["master_weights"] = True
        else:
            sharded_optim_index["master_weights"] = False

    if master_weights is None:
        return [(optim_state_dict, shard_optimizer_file, sharded_optim_index)]
    else:
        return [
            (optim_state_dict, shard_optimizer_file, sharded_optim_index),
            (master_weights, shard_master_weight_file, sharded_master_weight_index),
        ]


def check_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization=False):
    index_filename = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
    index_filename = os.path.join(resume_from_checkpoint, index_filename)

    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_weight_filenames = sorted(set(index["weight_map"].values()))

    existed_filelist = []
    existed_files = []
    for filename in os.listdir(resume_from_checkpoint):
        if filename in all_weight_filenames:
            existed_files.append(filename)

    dist.all_gather_object(existed_filelist, existed_files)
    flatten_existed_filelist = flatten_list(existed_filelist)
    diff_filelist = list(set(all_weight_filenames).difference(set(flatten_existed_filelist)))
    if len(diff_filelist) != 0:
        raise Exception(f"Sorry, the weight file list on the machines is not complete!, missing {diff_filelist}")

    resume_inplace = True
    if args.dataset_rank == 0:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
        pp_group = hcg.get_pipe_parallel_group()

        need_filelist = []
        for key in model.state_dict().keys():
            filename = index["weight_map"][key]
            if filename not in need_filelist:
                need_filelist.append(filename)
        diff_filelist = list(set(need_filelist).difference(set(existed_files)))
        num_diff = paddle.to_tensor([len(diff_filelist)])
        if tp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
        if pp_group.nranks > 1:
            dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
        if num_diff.item() == 0:
            resume_inplace = True
        else:
            resume_inplace = False
    resume_inplace = paddle.to_tensor([resume_inplace])
    dist.all_reduce(resume_inplace, op=dist.ReduceOp.PROD)
    resume_inplace = resume_inplace.item()
    return resume_inplace


def check_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    if not safe_serialization:
        index_filename, index_filename_master_weights = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    index_filename = os.path.join(resume_from_checkpoint, index_filename)
    index_filename_master_weights = os.path.join(resume_from_checkpoint, index_filename_master_weights)

    if distributed_isfile(index_filename):
        distributed_file(index_filename)
    else:
        raise Exception(
            f"Sorry, we can not find {index_filename}. This file should be appear at least on one machine."
        )

    with open(index_filename, "r") as f:
        index = json.loads(f.read())
    all_optimizer_filenames = sorted(set(index["weight_map"].values()))

    has_master_weights = index["master_weights"]
    if has_master_weights:
        if distributed_isfile(index_filename_master_weights):
            distributed_file(index_filename_master_weights)
        else:
            raise Exception(
                f"Sorry, we can not find {index_filename_master_weights}. This file should be appear at least on one machine."
            )
        with open(index_filename_master_weights, "r") as f:
            index_mw = json.loads(f.read())
        all_mw_filenames = sorted(set(index_mw["weight_map"].values()))

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    struct2static_name_mappings = {k: v.name for k, v in model.state_dict().items()}
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank

    def check_complete(all_filenames):
        existed_filelist = []
        existed_files = []
        for filename in os.listdir(resume_from_checkpoint):
            if filename in all_filenames:
                existed_files.append(filename)

        dist.all_gather_object(existed_filelist, existed_files)
        flatten_existed_filelist = flatten_list(existed_filelist)
        diff_filelist = list(set(all_filenames).difference(set(flatten_existed_filelist)))
        if len(diff_filelist) != 0:
            raise Exception(
                f"Sorry, the optimizer file list on `data_parallel_rank==0` machines is not complete!, missing {diff_filelist}"
            )
        return existed_files

    def check_dynamic_load(args, weight_map, existed_files, is_master_weights=False, typename_list=None):
        resume_inplace = True
        if args.data_parallel_rank == 0:
            need_filelist = []
            for key in model.state_dict().keys():
                if sharding_group.nranks > 1:
                    static_name = struct2static_name_mappings.get(key, None)
                    param_rank = param2rank.get(static_name, None)
                    if param_rank != sharding_rank:
                        continue

                if not is_master_weights:
                    for type_name in typename_list:
                        type_key = key + "/" + type_name
                        filename = weight_map[type_key]
                        if filename not in need_filelist:
                            need_filelist.append(filename)
                else:
                    filename = weight_map[key]
                    if filename not in need_filelist:
                        need_filelist.append(filename)

            diff_filelist = list(set(need_filelist).difference(set(existed_files)))
            num_diff = paddle.to_tensor([len(diff_filelist)])
            if tp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=tp_group)
            if pp_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=pp_group)
            if sharding_group.nranks > 1:
                dist.all_reduce(num_diff, op=dist.ReduceOp.MAX, group=sharding_group)

            if num_diff.item() == 0:
                resume_inplace = True
            else:
                resume_inplace = False
        resume_inplace = paddle.to_tensor([resume_inplace])
        dist.all_reduce(resume_inplace, op=dist.ReduceOp.PROD)
        return resume_inplace.item()

    # check whether the optimizer checkpoint files are complete.
    existed_files = check_complete(all_optimizer_filenames)
    if has_master_weights:
        existed_files_mw = check_complete(all_mw_filenames)
    # get optimizer's param type name, like moment1_0.
    typename_list = []
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        if typename not in typename_list:
            typename_list.append(typename)
    resume_inplace = check_dynamic_load(
        args, index["weight_map"], existed_files, is_master_weights=False, typename_list=typename_list
    )
    resume_inplace_rw = True
    if has_master_weights:
        resume_inplace_rw = check_dynamic_load(args, index_mw["weight_map"], existed_files_mw, is_master_weights=True)
    return resume_inplace & resume_inplace_rw


def create_dispatch_table(args, model, file_keyname_mappings, file_machine_mappings, resume_from_checkpoint):
    """Create dispatch table for dynamically loading state dict.

    Args:
        args
    """

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # create receive table.
    dispatch_list = []
    recv_table = {}
    if args.dataset_rank == 0:
        for (k, v) in model.state_dict().items():
            if hasattr(v, "is_distributed") and v.is_distributed:
                recv_table[k] = [(dist.get_rank(), tp_rank)]
            else:
                recv_table[k] = [(dist.get_rank(), -1)]

    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for key, value in dl.items():
            if key not in recv_table:
                recv_table[key] = value
            else:
                recv_table[key] += value

    # create send table.
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)

    return send_table, recv_table


def create_optimizer_dispatch_table(
    args,
    model,
    optimizer,
    file_keyname_mappings,
    file_machine_mappings,
    resume_from_checkpoint,
    struct2static_name_mappings,
    is_master_weights=False,
    typename_list=None,
):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    tp_rank = tp_group.rank

    # create receive table.
    dispatch_list = []
    recv_table = {}
    if args.data_parallel_rank == 0:
        for (k, v) in model.state_dict().items():
            if sharding_group.nranks > 1:
                static_name = struct2static_name_mappings[k]
                param_rank = param2rank.get(static_name, None)
                if param_rank != sharding_rank:
                    continue
            if is_master_weights:
                if hasattr(v, "is_distributed") and v.is_distributed:
                    recv_table[k] = [(dist.get_rank(), tp_rank)]
                else:
                    recv_table[k] = [(dist.get_rank(), -1)]
            else:
                for typename in typename_list:
                    type_key = k + "/" + typename
                    if "moment" in typename or "velocity" in typename:
                        if hasattr(v, "is_distributed") and v.is_distributed:
                            recv_table[type_key] = [(dist.get_rank(), tp_rank)]
                        else:
                            recv_table[type_key] = [(dist.get_rank(), -1)]
                    else:
                        recv_table[type_key] = [(dist.get_rank(), -1)]

    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for k, v in dl.items():
            if k not in recv_table:
                recv_table[k] = v
            else:
                recv_table[k] += v

    # create send table.
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)
    return send_table, recv_table


def dynamic_load_unified_checkpoint(args, model, resume_from_checkpoint, safe_serialization=False):

    index_filename = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
    index_filename = os.path.join(resume_from_checkpoint, index_filename)

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    send_table, recv_table = create_dispatch_table(
        args, model, file_keyname_mappings, file_machine_mappings, resume_from_checkpoint
    )

    all_tp_keys = []
    for k, v in recv_table.items():
        if k not in all_tp_keys:
            if v[0][1] != -1:
                all_tp_keys.append(k)

    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)
    state_dict = distributed_send_recv(
        config_revise,
        model.state_dict(),
        tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )

    error_msgs = _load_state_dict_into_model(model, state_dict, "")
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        raise RuntimeError(f"Error(s) in loading dynamic state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def dynamic_load_unified_optimizer(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    optim_state_dict = nested_copy(optimizer.state_dict())

    if safe_serialization:
        index_filename, index_filename_mw = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_mw = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME

    with open(os.path.join(resume_from_checkpoint, index_filename), "r") as f:
        index = json.loads(f.read())
    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    has_master_weights = index["master_weights"]
    if has_master_weights:
        with open(os.path.join(resume_from_checkpoint, index_filename_mw), "r") as f:
            index_mw = json.loads(f.read())
        file_keyname_mappings_mw, file_machine_mappings_mw = get_file_mappings(index_mw, resume_from_checkpoint)

    typename_list = []
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        if typename not in typename_list:
            typename_list.append(typename)
    struct2static_name_mappings = {k: v.name for k, v in model.state_dict().items()}
    send_table, recv_table = create_optimizer_dispatch_table(
        args,
        model,
        optimizer,
        file_keyname_mappings,
        file_machine_mappings,
        resume_from_checkpoint,
        struct2static_name_mappings,
        is_master_weights=False,
        typename_list=typename_list,
    )
    if has_master_weights:
        send_table_mw, recv_table_mw = create_optimizer_dispatch_table(
            args,
            model,
            optimizer,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
            resume_from_checkpoint,
            struct2static_name_mappings,
            is_master_weights=True,
        )

    # Initialize optimizer state dict.
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    optim_state_dict_mw = {}
    for k, v in model.state_dict().items():
        if sharding_group.nranks > 1:
            static_name = struct2static_name_mappings[k]
            param_rank = param2rank.get(static_name, None)
            if param_rank != sharding_group.rank:
                continue
        for typename in typename_list:
            new_k = k + "/" + typename
            if "beta" in typename:  # TODO: Is there a better way?
                optim_state_dict[new_k] = paddle.empty([1], dtype="float32")
            else:
                optim_state_dict[new_k] = paddle.empty_like(v, dtype="float32")
        if has_master_weights:
            optim_state_dict_mw[k] = paddle.empty_like(v, dtype="float32")

    all_tp_keys = []
    for k, v in recv_table.items():
        structure_name, typename = k.split("/")
        if "moment" in typename or "velocity" in typename:
            if structure_name not in all_tp_keys:
                if v[0][1] != -1:
                    all_tp_keys.append(structure_name)

    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)
    optimizer_keys = list(index["weight_map"].keys())
    optimizer_tp_actions = mapping_optimizer_tp_actions(tp_actions, optimizer_keys)
    if has_master_weights:
        optimizer_tp_actions.update(tp_actions)

    # Begin to send and recv optimizer tensor.
    optim_state_dict = distributed_send_recv(
        config_revise,
        optim_state_dict,
        optimizer_tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )
    if has_master_weights:
        optim_state_dict_mw = distributed_send_recv(
            config_revise,
            optim_state_dict_mw,
            optimizer_tp_actions,
            send_table_mw,
            recv_table_mw,
            resume_from_checkpoint,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
        )

    # rename optimizer.
    for key in list(optim_state_dict.keys()):
        if key == "LR_Scheduler":
            continue
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            key_name = "_".join([static_name, "fp32_master_0", key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        optim_state_dict[key_name] = optim_state_dict.pop(key)
        optim_state_dict[key_name].name = key_name

    if has_master_weights:
        optim_state_dict["master_weights"] = {}
        for key in list(optim_state_dict_mw.keys()):
            static_name = struct2static_name_mappings[key]
            optim_state_dict["master_weights"][static_name] = optim_state_dict_mw.pop(key)
            optim_state_dict["master_weights"][static_name].name = "_".join([static_name, "fp32_master_0"])

    if args.data_parallel_rank == 0:
        return optim_state_dict
    return None


def get_file_mappings(index, resume_from_checkpoint):
    file_keyname_mappings = {}
    for k, v in index["weight_map"].items():
        if v not in file_keyname_mappings:
            file_keyname_mappings[v] = []
        file_keyname_mappings[v].append(k)
    for k in file_keyname_mappings.keys():
        file_keyname_mappings[k] = sorted(file_keyname_mappings[k])

    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    global_rank = dist.get_rank()
    file_machine_mappings = {}
    for filename in file_keyname_mappings.keys():
        if local_rank == 0 and os.path.exists(os.path.join(resume_from_checkpoint, filename)):
            file_machine_mappings[filename] = [global_rank // local_device_count]
    file_machine_list = []
    dist.all_gather_object(file_machine_list, file_machine_mappings)
    file_machine_mappings = {}
    for mappings in file_machine_list:
        for k, v in mappings.items():
            if k not in file_machine_mappings:
                file_machine_mappings[k] = v
            else:
                file_machine_mappings[k] += v
    return file_keyname_mappings, file_machine_mappings


def create_send_table(file_keyname_mappings, file_machine_mappings):
    send_table = {}
    global_rank = dist.get_rank()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    for filename, keys in file_keyname_mappings.items():
        machine = file_machine_mappings[filename][0]
        is_src = (global_rank // local_device_count) == machine
        for i, key in enumerate(keys):
            if is_src and local_rank == i % local_device_count:
                send_table[key] = global_rank
    dispatch_list = []
    dist.all_gather_object(dispatch_list, send_table)
    send_table = {}
    for dl in dispatch_list:
        send_table.update(dl)
    return send_table


def distributed_send_recv(
    config,
    state_dict,
    tp_actions,
    send_table,
    recv_table,
    resume_from_checkpoint,
    file_keyname_mappings,
    file_machine_mappings,
):
    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    global_rank = dist.get_rank()
    for filename in file_keyname_mappings.keys():
        machine = file_machine_mappings[filename][0]
        is_src = global_rank // local_device_count == machine
        if is_src:
            f = safe_open(os.path.join(resume_from_checkpoint, filename), framework="np")

        for key in file_keyname_mappings[filename]:
            recv_info = recv_table[key]
            recv_ranklist = [a for (a, b) in recv_info]

            if is_src and global_rank == send_table[key]:
                py_safe_slice_ = f.get_slice(key)
                # send
                if key in tp_actions:
                    weight = tp_actions[key](py_safe_slice_)
                    # copy weight to GPU
                    for j in range(len(weight)):
                        with device_guard():
                            weight[j] = paddle.Tensor(weight[j], zero_copy=True)
                        weight[j] = weight[j]._copy_to(paddle.framework._current_expected_place(), False)

                    for recv_rank, split_index in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight[split_index]
                        else:
                            dist.stream.send(weight[split_index], dst=recv_rank)
                else:
                    # no need to tp split
                    weight = py_safe_slice_[:]
                    with device_guard():
                        weight = paddle.Tensor(weight, zero_copy=True)
                    weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                    for recv_rank, _ in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight
                        else:
                            dist.stream.send(weight, dst=recv_rank)

            if global_rank != send_table[key] and global_rank in recv_ranklist:
                dist.stream.recv(state_dict[key], src=send_table[key])

        if is_src:
            f.__exit__(None, None, None)

    return state_dict


def get_sharded_file_name(args, file_name, is_optimizer=False):
    if not is_optimizer:
        shard_file = file_name.replace(
            ".pdparams",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.pdparams",
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//args.dataset_world_size:05d}.safetensors",
        )
    else:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        shard_file = file_name.replace(
            ".pdparams", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdparams"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.safetensors",
        )
        shard_file = shard_file.replace(
            ".pdopt", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//dp_group.nranks:05d}.pdopt"
        )
    return shard_file


def get_sharded_index(
    index_file_list,
    total_size_list,
):
    # save index json file
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    if local_rank == 0:
        sharded_index_json = {}

        sharded_index_json["metadata"] = {"total_size": sum(total_size_list)}

        weight_map = {}
        for i, index_file in enumerate(index_file_list):
            weight_map.update(index_file_list[i])

        sharded_index_json["weight_map"] = weight_map
        return sharded_index_json

    return None


def gather_sharded_object(index_file, total_size, is_optimizer=False):

    index_file_list, total_size_list = [], []

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    logger.info("Unified checkpoint generating sharded_index json files.")

    if tp_group.nranks > 1:
        dist.all_gather_object(index_file_list, index_file, tp_group)
        dist.all_gather_object(total_size_list, total_size, tp_group)
    if pp_group.nranks > 1:
        pp_index_file_list = []
        pp_total_size_list = []
        dist.all_gather_object(
            pp_index_file_list, index_file_list if len(index_file_list) > 0 else index_file, pp_group
        )
        dist.all_gather_object(
            pp_total_size_list, total_size_list if len(total_size_list) > 0 else total_size, pp_group
        )
        index_file_list = pp_index_file_list
        total_size_list = pp_total_size_list

    index_file_list = flatten_list(index_file_list)
    total_size_list = flatten_list(total_size_list)

    # for pure sharding
    if len(index_file_list) == 0 and len(total_size_list) == 0:
        index_file_list = [index_file]
        total_size_list = [total_size]
    if is_optimizer:
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_group.nranks > 1:
            sharding_index_file_list = []
            sharding_total_size_list = []
            dist.all_gather_object(sharding_index_file_list, index_file_list, sharding_group)
            dist.all_gather_object(sharding_total_size_list, total_size_list, sharding_group)
            index_file_list = flatten_list(sharding_index_file_list)
            total_size_list = flatten_list(sharding_total_size_list)

    return index_file_list, total_size_list


def generate_base_static_name(vname):
    # return base static name and specific type name, like [embedding_0.w_0, moment1_0]
    if "fp32_master_0" in vname:
        vname = vname.split("_fp32_master_0_")
        return vname[0], vname[1]
    else:
        vname = vname.split(".")
        a = vname[0] + "." + vname[1][:3]
        b = vname[1][4:]
        return a, b


def filter_params(model_to_save, state_dict, is_optimizer=False):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # for pure sharding or pure pp
    if tp_size <= 1:
        return [list(state_dict.keys())]

    filter_tensor_list = [[] for i in range(tp_size)]

    if tp_rank == 0:
        tensor_bytes_dict = {}
        model_state_dict = model_to_save.state_dict()
        for (k, v) in state_dict.items():
            model_v = model_state_dict[k.split("/")[0]] if is_optimizer else v
            if hasattr(model_v, "is_distributed") and model_v.is_distributed:
                tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
            else:
                tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)

        filter_tensor_list = []
        current_block = []
        current_block_size = 0
        total_size = 0

        max_shard_size = (sum(tensor_bytes_dict.values()) + tp_size - 1) // tp_size

        for index, (key, weight_size) in enumerate(tensor_bytes_dict.items()):
            # If this weight is going to tip up over the maximal size, we split.
            # if current_block_size + weight_size > max_shard_size:
            if total_size + weight_size > max_shard_size * (len(filter_tensor_list) + 1) or (
                len(tensor_bytes_dict) - index < (tp_size - len(filter_tensor_list))
            ):
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    filter_tensor_list.append(current_block)
                current_block = []
                current_block_size = 0

            current_block.append(key)
            current_block_size += weight_size
            total_size += weight_size

        filter_tensor_list.append(current_block)
        assert len(filter_tensor_list) == tp_size, "Error, partition failed!"

    dist.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    return filter_tensor_list


def merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys):
    logger.info("Unified checkpoint merge tensor parallel in shards")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
    if hcg.get_pipe_parallel_group().nranks > 1:
        filter_keys = set([y for x in all_filter_keys for y in x])
        for key in list(tp_actions.keys()):
            if key not in filter_keys:
                tp_actions.pop(key)

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            key = filter_keys[i]
            tensor = state_dict[key]
            if key in tp_actions:
                ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                action = tp_actions.pop(key)
                tensor = action(ret) if is_dst else None
            else:
                tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

            if is_dst:
                state_dict_to_save[key] = tensor

    if len(tp_actions) > 0:
        for x in tp_actions.keys():
            logger.warning(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

    return state_dict_to_save


def merge_tensor_parallel_for_optimizer(state_dict, tp_actions, all_filter_keys):
    logger.info("Unified optimizer tensor parallel in shards")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            # get base model key
            model_key = filter_keys[i].split("/")[0]
            tensor = state_dict[filter_keys[i]]
            if model_key in tp_actions:
                # for example: beta1, beta2
                if tensor.numel().item() == 1:
                    tensor = (
                        tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None
                    )  # Need broadcast when loaded
                else:
                    ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                    action = tp_actions[model_key]
                    tensor = action(ret) if is_dst else None
            else:
                tensor = tensor._copy_to(paddle.CPUPlace(), False) if is_dst else None

            if is_dst:
                state_dict_to_save[filter_keys[i]] = tensor

    return state_dict_to_save


def get_optimizer_shard_files(optimizer_path, index_filename):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a optimizer index ({index_filename}) in {optimizer_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_optimizer_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    sharded_metadata["master_weights"] = index.get("master_weights", False)

    file_map = {file: set() for file in shard_filenames}
    for weight, file in index["weight_map"].items():
        file_map[file].add(weight)

    sharded_metadata["file_map"] = file_map

    # First, let's deal with local folder.
    # TODO: if optimizer_path is a folder, we should check if the optimizer is already cached or not.
    if os.path.isdir(optimizer_path):
        shard_filenames = [os.path.join(optimizer_path, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata


def get_expected_keys(sharded_metadata, model, optimizer):
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    in_sharding_parallel_model = sharding_group.nranks > 1
    if in_sharding_parallel_model:
        params2rank = optimizer._param2rank

    struct2static_name_mappings = {k: v.name for k, v in model.state_dict().items()}

    expected_keys = []
    for key in list(sharded_metadata["all_optimizer_keys"]):
        key_name = key.split("/")[0]
        static_name = struct2static_name_mappings.get(key_name, None)

        if in_sharding_parallel_model:
            params_rank = params2rank.get(static_name, None)
            if params_rank == sharding_rank:
                expected_keys.append(key)
        else:
            if static_name is not None:
                expected_keys.append(key)
    expected_keys = set(expected_keys)

    loaded_keys = sharded_metadata["all_optimizer_keys"]
    missing_keys = expected_keys - set(loaded_keys)
    if len(missing_keys) > 0:
        raise ValueError(f"optimizer missing weights keys: {missing_keys}")

    return expected_keys


def mapping_optimizer_tp_actions(tp_actions, optimizer_loaded_keys):
    """# convert param.name to
    param.key/moment1_0
    or param.key/beta1_XXX
    or param.key/beta2_XXX
    Args:
        tp_actions (dict): dictionay of tensor parallel actions {key: action}
        optimizer_loaded_keys (list or set): [param.key1/moment1_0, param.key2/beta1_XXX, param.key3/beta2_XXX]
    Returns:
        dict: new dictionay of tensor parallel actions {key: action}
    """
    new_actions = {}
    for key in optimizer_loaded_keys:
        key_base = key.split("/")[0]
        if ("moment" in key.split("/")[1] or "velocity" in key.split("/")[1]) and key_base in tp_actions:
            new_actions[key] = tp_actions[key_base]
    return new_actions


def nested_copy(inputs):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy(inputs[key])
        return outputs
    return inputs


def nested_copy_place(inputs, place=None):
    if isinstance(inputs, dict):
        outputs = {}
        for key in list(inputs.keys()):
            outputs[key] = nested_copy_place(inputs[key], place)
        return outputs
    if isinstance(inputs, paddle.Tensor):
        inputs = inputs._copy_to(place, False)

    return inputs


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list
