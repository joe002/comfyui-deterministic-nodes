# ComfyUI Deterministic Nodes
# Copyright (C) 2025 Joseph Ibrahim <joseph@josephibrahim.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing available: See COMMERCIAL_LICENSE.md

"""
ComfyUI Deterministic Nodes
===========================

Batch-invariant inference nodes for guaranteed reproducibility.

Based on:
- ThinkingMachines batch-invariant-ops research
- NVIDIA CES 2026 multi-model agent patterns
- Ralph v3 file-centric state management
- ECHO 2.0 4-tier context memory

Nodes:
- DeterministicSampler: Batch-invariant sampling (batch_size=1 per item)
- ChecksumValidator: Reproducibility proof with tolerance modes
- MoERouterNode: Deterministic expert routing (hash-based)
- CascadeRefiner: Sequential 3-stage refinement (Nemotron pattern)
- ECHOContextNode: 4-tier context memory retrieval
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
