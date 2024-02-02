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

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class ReturnValueWithError(BuildingBlock):
    """
    calculate system call return value for all syscalls.
    Returns:
        - return value as is if it is an integer
        - Error code if it is an error
        - 0 if it is not an integer and not an error
    """

    def __init__(self):
        super().__init__()

    def _calculate(self, syscall: Syscall):
        return_value_string = syscall.param('res')
        if return_value_string is not None:
            try:
                return_value = int(return_value_string)
            except ValueError:
                if '(E' in return_value_string:
                    return_value = return_value_string
                else:
                    return_value = 0
            return return_value

    def depends_on(self):
        return []
