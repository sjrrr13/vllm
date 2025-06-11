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
import multiprocessing
import weakref
from typing import Any, Optional

from vllm.logger import init_logger
from vllm.entrypoints.cli.serve import run_api_server_worker_proc
from vllm.v1.utils import shutdown

logger = init_logger(__name__)


class ServerActor:
    """Manages API server as ray actor.
    
    Handles creation, monitoring, and termination of API server. 
    """

    def __init__(
        self,
        listen_address: str,
        sock: Any,
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

        spawn_context = multiprocessing.get_context("spawn")
        proc = spawn_context.Process(target=run_api_server_worker_proc,
                                     name=f"ApiServer_{client_index}",
                                     args=(listen_address, sock, args,
                                           client_config))
        self.processes.append(proc)
        proc.start()
        
        logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def close(self) -> None:
        self._finalizer()
