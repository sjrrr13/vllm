# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import ray
from ray.util.placement_group import PlacementGroup

from vllm.logger import init_logger
from vllm.config import ParallelConfig

logger = init_logger(__name__)

WAIT_PLACEMENT_GROUP_TIMEOUT: float = 5.0


def initialize_placement_group(
    placement_group_name: str,
    num_cpus: int,
    num_gpus: int,
    parallel_config: ParallelConfig,
    detached: bool = False,
    block: bool = True,
    node_id: str = None,
    resources: Dict[str, float] = {},
) -> PlacementGroup:
    """Initialize the distributed cluster probably with Ray.

    Args:
        placement_group_name: The name of placement group.
        num_cpus: The number of cpus in placement group.
        num_gpus: The number of gpus in placement group.
        gpu_bundling_strategy: GPU bundle st.
        detached: Whether the lifetime of the placement group being detached.
        block: If True, the function will block until the placement group 
               is ready.
        node_id: The node id of node. If specified, placement group will be 
                 created on the specified node.
        resources: The addtional resources requirements of placement group.

    Returns:
        `placement_group`. `placement_group` includes the specification
        of the resources for each distributed worker.
    """
    if ray is None:
        raise ImportError(
            "Ray is not installed. Please install Ray to use distributed "
            "serving.")

    lifetime = "detached" if detached else None

    num_gpus_in_cluster = ray.cluster_resources().get("GPU", 0)
    if num_gpus > num_gpus_in_cluster:
        raise ValueError(
            f"The number of required GPUs {num_gpus} exceeds the total number"
            f" of available GPUs {num_gpus_in_cluster} in the cluster.")
    
    dp_master_ip = parallel_config.data_parallel_master_ip
    dp_size = parallel_config.data_parallel_size
    world_size = parallel_config.world_size
    local_dp_ranks: List[int] = []

    try:
        # Create a new placement group
        if dp_size > 1:
            if num_cpus < 2 * dp_size + 1 or num_gpus < world_size * dp_size:
                raise ValueError(
                    "The number of required CPUs/GPUs for data parallelism "
                    "exceeds the total number of available in the cluster.")
            placement_group_specs = \
                [{"CPU": 1, "driver_node": 1}] + \
                [{"CPU": 2, "GPU": world_size}] * dp_size

        elif num_gpus >= 1:
            # bundle_0: All CPU Actors + Worker_0, 
            # bundle_1-N-1: Worker_1...Worker_N-1
            placement_group_specs = [{"CPU": num_cpus, "GPU": 1}] + \
                                    [{"GPU": 1}] * (num_gpus - 1)
        else:
            placement_group_specs = [{"CPU": num_cpus}]
        if resources:
            placement_group_specs += [resources]
        # pylint: disable=self-assigning-variable
        placement_group_specs = (placement_group_specs)

        logger.debug(f"placement_group_specs: {placement_group_specs}")

        # PACK (not STRICT_PACK) to support multi-node placement group.
        if node_id is None:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "PACK", 
                name=placement_group_name, lifetime=lifetime)
        else:
            current_placement_group = ray.util.placement_group(
                placement_group_specs, "STRICT_PACK", 
                name=placement_group_name, lifetime=lifetime, 
                _soft_target_node_id=node_id)
        # Wait until PG is ready - this will block until all
        # requested resources are available, and will timeout
        # if they cannot be provisioned.
        if block:
            try:
                ray.get(current_placement_group.ready(), 
                        timeout=WAIT_PLACEMENT_GROUP_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.warning(f"Waiting for new placement group "
                               f"{placement_group_name} ready timeout.")
                return None
    except Exception as e: # pylint: disable=broad-except
        logger.exception(f"Error during initialize placement group "
                         f"{placement_group_name}, unexpected exception: {e}")
        return None

    local_dp_ranks = [0, 1]
    return current_placement_group, local_dp_ranks


def wait_for_completion_or_failure(
        api_server_manager: APIServerProcessManager,
        engine_manager: Optional[Union[CoreEngineProcManager,
                                       CoreEngineActorManager]] = None,
        coordinator: Optional["DPCoordinator"] = None) -> None:
    try:
        logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        sentinel_to_proc: dict[Any, BaseProcess] = {
            proc.sentinel: proc
            for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        actor_run_refs = []
        if isinstance(engine_manager, CoreEngineProcManager):
            for proc in engine_manager.processes:
                sentinel_to_proc[proc.sentinel] = proc
        elif isinstance(engine_manager, CoreEngineActorManager):
            actor_run_refs = engine_manager.get_run_refs()

        # Check if any process terminates
        while sentinel_to_proc or actor_run_refs:
            # Wait for any process to terminate
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc,
                                                         timeout=5)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}")

            if actor_run_refs:
                import ray
                _, actor_run_refs = ray.wait(actor_run_refs, timeout=5)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s",
                         str(e))
        raise
    finally:
        logger.info("Terminating remaining processes ...")
        api_server_manager.close()
        if coordinator:
            coordinator.close()
        if engine_manager:
            engine_manager.close()
