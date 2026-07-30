"""The suite, run against its own controls.

`MemoryHarness` is the positive control and must pass everything at `METADATA`: a case it fails
is a case asserting something untrue of Polars itself, which makes it the suite's bug and not a
harness's. Parquet and IPC are real formats, and their declared losses are the ones a plugin
author can attribute to the format rather than to themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plioc import ConformanceSuite
from plioc.harnesses.files import IpcHarness, ParquetHarness
from plioc.harnesses.memory import MemoryHarness
from plioc.harnesses.plugin import PluginHarness


class TestMemory(ConformanceSuite):
    @pytest.fixture
    def harness(self) -> MemoryHarness:
        return MemoryHarness()


class TestPlugin(ConformanceSuite):
    @pytest.fixture
    def harness(self) -> PluginHarness:
        return PluginHarness()


class TestParquet(ConformanceSuite):
    @pytest.fixture
    def harness(self, tmp_path: Path) -> ParquetHarness:
        return ParquetHarness(tmp_path)


class TestIpc(ConformanceSuite):
    @pytest.fixture
    def harness(self, tmp_path: Path) -> IpcHarness:
        return IpcHarness(tmp_path)
