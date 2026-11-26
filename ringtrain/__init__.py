"""ringtrain: data- and pipeline-parallel training over hand-written TCP collectives, with
gradient compression and fault-tolerant checkpoints in raftstore."""
from .collectives import ProcessGroup, chunk_bounds, naive_allreduce
from .transport import Endpoint, GenerationAborted, PeerFailure, Transport

__version__ = "0.1.0"
__all__ = ["ProcessGroup", "Transport", "Endpoint", "PeerFailure", "GenerationAborted",
           "chunk_bounds", "naive_allreduce", "__version__"]
