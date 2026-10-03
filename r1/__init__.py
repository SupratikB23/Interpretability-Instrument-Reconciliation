"""Top-level alias so ``python -m r1.run`` works.

The implementation lives in ``src/r1/run.py``; this shim keeps the documented
command line intact without duplicating logic.
"""

from src.r1.run import STAGE_DISPATCH, STAGES, RunContext, build_parser, main

__all__ = ["STAGES", "STAGE_DISPATCH", "RunContext", "build_parser", "main"]
