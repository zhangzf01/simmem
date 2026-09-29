"""Cross-step user-simulator memory.

The memory is a list of notes appended to the persuadee system prompt. After every training step,
groups whose reward is high and whose std is low are sent to a judge; when the judge says the
persuader got there by exploiting the simulator, the reason becomes a new note for later steps.
"""
from .config import SimMemSettings, add_arguments
from .updater import SimMemUpdater, get_updater

__all__ = ["SimMemSettings", "add_arguments", "SimMemUpdater", "get_updater"]
