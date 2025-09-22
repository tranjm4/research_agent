import pytest
import unittest
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from server.mcp_server import mcp

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMCPServer:
    pass