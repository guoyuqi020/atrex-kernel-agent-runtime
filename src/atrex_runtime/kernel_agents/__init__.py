"""Full-repository Kernel Agent Bundle manifests and builders."""

from .git import (
    GitOptimizerBaseLoader,
    GitOptimizerBaseResult,
    OptimizerSourceProvenanceV1,
    OptimizerSubmoduleProvenanceV1,
)
from .revision import (
    KERNEL_AGENT_BUNDLE_FORMAT,
    KERNEL_AGENT_BUNDLE_MANIFEST,
    KERNEL_AGENT_BUNDLE_MANIFEST_VERSION,
    KernelAgentBundleEntrypointV1,
    KernelAgentBundleLimits,
    KernelAgentBundleManifestV1,
    KernelAgentRevisionBuilder,
    is_ignored_kernel_agent_path,
)

__all__ = [
    "KERNEL_AGENT_BUNDLE_FORMAT",
    "KERNEL_AGENT_BUNDLE_MANIFEST",
    "KERNEL_AGENT_BUNDLE_MANIFEST_VERSION",
    "GitOptimizerBaseLoader",
    "GitOptimizerBaseResult",
    "KernelAgentBundleEntrypointV1",
    "KernelAgentBundleLimits",
    "KernelAgentBundleManifestV1",
    "KernelAgentRevisionBuilder",
    "OptimizerSourceProvenanceV1",
    "OptimizerSubmoduleProvenanceV1",
    "is_ignored_kernel_agent_path",
]
