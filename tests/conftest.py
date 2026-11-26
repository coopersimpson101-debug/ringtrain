import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def raftstore_cluster():
    """A real 3-process raftstore cluster on localhost (nas, pi5, server)."""
    devcluster = pytest.importorskip("raftstore.devcluster", reason="raftstore not installed "
                                     "(pip install 'raftstore @ git+https://github.com/coopersimpson101-debug/raftstore')")
    work = tempfile.mkdtemp(prefix="ringtrain-rs-")
    dc = devcluster.DevCluster(work, names=("nas", "pi5", "server"), tick_ms=50)
    dc.start_all()
    try:
        dc.wait_leader(90)
        yield dc
    finally:
        dc.stop_all()
