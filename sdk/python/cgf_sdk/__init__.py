"""
CGF SDK - HostAdapter Python Client
Version: 0.1.0
"""

__version__ = "0.1.0"
__all__ = ["CGFClient", "HostAdapter", "GovernanceError", "FailModeError"]

from .cgf_client import CGFClient
from .adapter_base import HostAdapter
from .errors import GovernanceError, FailModeError
