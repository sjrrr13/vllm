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

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.utils import EngineZmqAddresses
from vllm.v1.engine.core import DPEngineCoreActor
from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


class DPLlumletActor:
    """Ray actor for running EngineCore in a data parallel context.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        # vLLM engine core
        logger.info("try to pull up DPEngineCoreActor")
        self.backend_engine = DPEngineCoreActor(
            vllm_config, on_head_node, addresses, executor_class,
            log_stats, dp_rank, local_dp_rank)
        
        logger.info(f"DPEngineCoreActor-{dp_rank} set up")
        
    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()

    def get_node_ip(self):
        node_id = ray.get_runtime_context().get_node_id()
        nodes = ray.nodes()
        for node in nodes:
            if node["NodeID"] == node_id:
                return node["NodeManagerAddress"]
        return None
