#  LID-DS-TF Copyright (c) 2024. Tinsaye Abye
#
#  LID-DS-TF is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  LID-DS-TF is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with LID-DS-TF.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional

from algorithms.building_block import BuildingBlock
from algorithms.features.impl.filedescriptor import FileDescriptor, FDMode
from dataloader.syscall import Syscall


class PathLikeParam(BuildingBlock):
    """
    Returns the first parameter that exists given a list of possible parameters.
    Path like parameters can be:
        - fd
        - in_fd
        - out_fd
        - path
        - name
        - oldpath
        - newpath
        - filename
        - exe
    This is a very simple implementation, it does not check if the parameter is a path.
    """

    def __init__(self, params: list[str]):
        super().__init__()
        self._params = params

    def _calculate(self, syscall: Syscall) -> Optional[tuple]:
        for param in self._params:
            path = syscall.param(param)
            if path is not None:
                if param in ["fd", "in_fd", "out_fd"]:
                    if "<f>" in path:
                        # noinspection PyProtectedMember
                        result = FileDescriptor._get_fd_part(path, FDMode.Content)
                        return result
                else:
                    return path,

    def depends_on(self) -> list:
        return []
