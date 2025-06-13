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

import argparse
import copy

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import vllm.envs as envs
from vllm import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.dp_manager.coordinator_actor import DPCoordinatorActor
from vllm.v1.executor.abstract import Executor
from vllm.v1.dp_manager.llumlet_actor import DPLlumletActor
from vllm.v1.engine.core import DPEngineCoreActor
from vllm.v1.dp_manager.server_actor import ServerActor
from vllm.v1.dp_manager.utils import (initialize_placement_group, 
                                      wait_for_completion_or_failure)
from vllm.v1.utils import (EngineZmqAddresses, get_engine_client_zmq_addr)

logger = init_logger(__name__)


class DPManager:
    """Manager actor used for data-parallel deployments (DP>1).

    This actor acts as the centralized component for dp instance 
    pull-up, coordination and fault tolerance.

    * Both the pull-up and handshake of ServerActor and Llumlet 
      are done within the DPManager constructor, so that once 
      the DPManager is successfully pulled up, the entire EP 
      unit is completely ready.

    * The DPManager is also responsible for the fault tolerance 
       of the entire ep unit deployment.
    
    """
    def __init__(self, args: argparse.Namespace):
        # Process args
        num_api_servers = args.api_server_count
        assert num_api_servers > 0

        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(
            usage_context=usage_context)
        model_config = vllm_config.model_config

        if num_api_servers > 1:
            if not envs.VLLM_USE_V1:
                raise ValueError("api_server_count > 1 is only supported "
                                 "for V1")

            if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
                raise ValueError("VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot "
                                "be used with api_server_count > 1")

            if model_config.is_multimodal_model and not (
                    model_config.disable_mm_preprocessor_cache):
                logger.warning(
                    "Multi-model preprocessor cache will be disabled for "
                    "api_server_count > 1")
                model_config.disable_mm_preprocessor_cache = True

        parallel_config = vllm_config.parallel_config
        assert parallel_config.data_parallel_rank == 0

        dp_size = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local
        host = parallel_config.data_parallel_master_ip
        local_only = local_engine_count == dp_size

        # Set up ray
        if ray.is_initialized():
            logger.info(
                "Ray is already initialized. Skipping Ray initialization")
        else:
            ray.init(resources={"driver_node": 1})

        # TODO: set using args
        num_cpus = 64
        num_gpus = 4
        # TODO: test without DP
        pg, local_dp_ranks = initialize_placement_group(
            "pg", num_cpus, num_gpus, parallel_config)

        # Set up input and output addresses.
        input_addresses = [
            get_engine_client_zmq_addr(local_only, host)
            for _ in range(num_api_servers)
        ]
        output_addresses = [
            get_engine_client_zmq_addr(local_only, host)
            for _ in range(num_api_servers)
        ]

        addresses = EngineZmqAddresses(
            inputs=input_addresses,
            outputs=output_addresses,
        )

        coordinator = None
        stats_update_address = None

        # Pull-up DPCoordinator for DP>1
        if dp_size > 1:
            assert num_api_servers == dp_size, \
                "Number of API Server should be equal with dp size"
            
            coordinator = ray.remote(DPCoordinatorActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0)
            ).remote(parallel_config)
            ray.get(pg.ready())
            # TODO: break coordinator's link
            addresses.coordinator_input, addresses.coordinator_output = \
                ray.get(coordinator.get_engine_socket_addresses.remote())
            stats_update_address = \
                ray.get(coordinator.get_stats_publish_address.remote())
            logger.info(f"Started DPCoordinator actor at "
                        f"{ray.get(coordinator.get_node_ip.remote())}")
            
        llumlet_actors = []
        server_actors = []
        executor_class = Executor.get_class(vllm_config)
        log_stats = not engine_args.disable_log_stats
        local_engine_count = \
            vllm_config.parallel_config.data_parallel_size_local

        for i in range(dp_size):
            dp_vllm_config = copy.deepcopy(vllm_config)
            on_head_node = i < local_engine_count
            # TODO: local_index (local_dp_rank)
            local_index = local_dp_ranks[i]

            # Pull-up Llumlet actor
            llumlet_actor = ray.remote(DPLlumletActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i+1,
                )).remote(vllm_config=dp_vllm_config,
                          executor_class=executor_class,
                          log_stats=log_stats,
                          on_head_node=on_head_node,
                          addresses=addresses,
                          dp_rank=i,
                          local_dp_rank=local_index)
            llumlet_actors.append(llumlet_actor.wait_for_init.remote())
        ray.get(llumlet_actors)

        logger.info("Llumlet actors init finished")
            
        for i in range(dp_size):
            # Pull-up ServerActor
            logger.info(f"try init ServerActor {i}")
            server_actor = ray.remote(ServerActor).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i+1)
            ).remote(
                args=args,
                input_address=input_addresses[i],
                output_address=output_addresses[i],
                stats_update_address=stats_update_address,
                client_index=i
            )
            server_actors.append(server_actor)

        # for i in range(dp_size):
            # logger.info(f"Started Llumlet actor {i} at "
            #             f"{ray.get(llumlet_actors[i].get_node_ip.remote())}")
            # logger.info(f"Started Server actor {i} at "
            #             f"{ray.get(server_actor[i].get_node_ip.remote())}")

        # Check initialization completion
        wait_for_completion_or_failure()

        logger.info("DPManager Launche finished")

        return
