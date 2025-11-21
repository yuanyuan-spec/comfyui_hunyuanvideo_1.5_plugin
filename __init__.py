# Copyright (C) 2025 yuanyuan-spec
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .nodes import NODE_CLASS_MAPPINGS as NODES_CLASS, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY
from .node_adv import NODE_CLASS_MAPPINGS as ADV_NODES_CLASS, NODE_DISPLAY_NAME_MAPPINGS as ADV_NODES_DISPLAY
from .node_sr import NODE_CLASS_MAPPINGS as SR_NODES_CLASS, NODE_DISPLAY_NAME_MAPPINGS as SR_NODES_DISPLAY

NODE_CLASS_MAPPINGS = {**NODES_CLASS, **ADV_NODES_CLASS,**SR_NODES_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY, **ADV_NODES_DISPLAY,**SR_NODES_DISPLAY}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]