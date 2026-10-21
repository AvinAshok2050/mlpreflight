"""preflightml - run this before you train your classifier."""
from .api import inspect
from .report import Report
from .core.finding import Finding, Severity

__version__ = "0.1.0"
__all__ = ["inspect", "Report", "Finding", "Severity"]
