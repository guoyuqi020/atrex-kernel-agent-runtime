"""Trusted control plane for self-evolving Atrex Kernel Agents."""

from .composition.campaign import CampaignRuntime, build_campaign_runtime
from .controller.campaign import CampaignScheduler, CampaignScheduleResult
from .controller.epoch import EpochController, EpochRunResult
from .registry.sqlite import SqliteRegistry

__all__ = [
    "CampaignRuntime",
    "CampaignScheduleResult",
    "CampaignScheduler",
    "EpochController",
    "EpochRunResult",
    "SqliteRegistry",
    "build_campaign_runtime",
]
