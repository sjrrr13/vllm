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

import ray

from vllm.v1.engine.coordinator import DPCoordinator


class DPCoordinatorActor(DPCoordinator):
    def get_node_ip(self):
        node_id = ray.get_runtime_context().get_node_id()
        nodes = ray.nodes()
        for node in nodes:
            if node["NodeID"] == node_id:
                return node["NodeManagerAddress"]
        return None

# import weakref

# import ray
# from ray.util.placement_group import PlacementGroup
# from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# from vllm.config import ParallelConfig
# from vllm.v1.engine.coordinator import CoordinatorProc
# from vllm.logger import init_logger
# from vllm.utils import get_open_zmq_ipc_path
# from vllm.v1.utils import get_engine_client_zmq_addr

# logger = init_logger(__name__)


# class DPCoordinatorActor:
#     """Pull up busy loop of DPCoordinator as ray task.
#     """
    
#     def __init__(self, parallel_config: ParallelConfig, 
#                  placement_group: PlacementGroup):
#         # Assume coordinator is colocated with front-end procs.
#         front_publish_address = get_open_zmq_ipc_path()

#         dp_size = parallel_config.data_parallel_size
#         assert dp_size > 1, "Coordinator only used for data parallel"

#         local_only = dp_size == parallel_config.data_parallel_size_local
#         host = parallel_config.data_parallel_master_ip
#         back_publish_address = get_engine_client_zmq_addr(local_only, host)
#         back_output_address = get_engine_client_zmq_addr(local_only, host)

#         self.proc = ray.remote(run_coordinator).options(
#             scheduling_strategy=PlacementGroupSchedulingStrategy(
#                 placement_group=placement_group,
#                 placement_group_bundle_index=0
#             )
#         ).remote(
#             parallel_config.data_parallel_size,
#             front_publish_address,
#             back_output_address,
#             back_publish_address,
#         )

#         self.stats_publish_address = front_publish_address
#         self.coord_in_address = back_publish_address
#         self.coord_out_address = back_output_address
        
#         self._finalizer = weakref.finalize(self, self._cleanup)

#     def get_stats_publish_address(self) -> str:
#         return self.stats_publish_address

#     def get_engine_socket_addresses(self) -> tuple[str, str]:
#         """Returns tuple of ZMQ input address, output address."""
#         return self.coord_in_address, self.coord_out_address

#     def _cleanup(self):
#         if hasattr(self, 'proc'):
#             ray.cancel(self.proc)
#             try:
#                 ray.get(self.proc)
#             except ray.exceptions.RayTaskError:
#                 pass

#     def close(self):
#         self._finalizer()


# def run_coordinator(
#         engine_count: int,
#         front_publish_address: str,
#         back_output_address: str,
#         back_publish_address: str,
#     ):
#         coordinator = CoordinatorProc(engine_count=engine_count)
#         try:
#             coordinator.process_input_socket(
#                 front_publish_address,
#                 back_output_address,
#                 back_publish_address,
#             )
#         except KeyboardInterrupt:
#             logger.info("DP Coordinator process exiting")
