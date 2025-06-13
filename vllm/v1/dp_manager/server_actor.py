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

import os
import sys
import argparse
import uvloop
import multiprocessing
import weakref
from typing import Any, Optional

import ray

from vllm.logger import init_logger
from vllm.entrypoints.openai.api_server import setup_server
from vllm.v1.utils import shutdown
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.entrypoints.openai.api_server import run_server_worker

logger = init_logger(__name__)


def run_api_server_worker_proc(listen_address,
                               sock,
                               args,
                               client_config=None,
                               **uvicorn_kwargs) -> None:
    """Entrypoint for individual API server worker processes."""

    # Add process-specific prefix to stdout and stderr.
    from multiprocessing import current_process
    process_name = current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config,
                          **uvicorn_kwargs))


class ServerActor:
    """Manages API server as ray actor.
    
    Handles creation, monitoring, and termination of API server. 
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_address: list[str],
        output_address: list[str],
        client_index: int = 0,
        stats_update_address: Optional[str] = None,
    ):
        """Initialize and start API server.
        
        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address 
        """
        
        listen_address, sock = setup_server(args)
        
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # Start API servers
        client_config = {
            "input_address": input_address,
            "output_address": output_address,
            "client_index": client_index
        }
        if stats_update_address is not None:
            client_config["stats_update_address"] = stats_update_address

        # logger.info("try get spawn_context")
        spawn_context = multiprocessing.get_context("spawn")
        logger.info("try launch Process")
        proc = spawn_context.Process(target=run_api_server_worker_proc,
                                     name=f"ApiServer_{client_index}",
                                     args=(listen_address, sock, args,
                                           client_config))
        proc.start()
        
        logger.info(f"Started API server processes {client_index}")

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def close(self) -> None:
        self._finalizer()

    def get_node_ip(self):
        node_id = ray.get_runtime_context().get_node_id()
        nodes = ray.nodes()
        for node in nodes:
            if node["NodeID"] == node_id:
                return node["NodeManagerAddress"]
        return None
