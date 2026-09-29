"""
Unit test for PySide6 Mission-Control Dashboard initialization.
"""

import os
import sys
import pytest

# Use offscreen platform for headless test execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from app.gui.dashboard import FSOCMissionControlDashboard


def test_dashboard_instantiation():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dashboard = FSOCMissionControlDashboard()
    assert dashboard is not None
    assert dashboard.windowTitle().startswith("SIH 2026")

    # Verify key widgets exist
    assert dashboard.camera_view is not None
    assert dashboard.state_banner is not None
    assert dashboard.plots is not None
    assert dashboard.panel_dist is not None
    assert dashboard.panel_pid is not None

    # Clean up worker thread
    dashboard.worker.running = False
    dashboard.worker.wait(500)
    dashboard.close()
