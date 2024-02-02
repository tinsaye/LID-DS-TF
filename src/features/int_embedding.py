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


class IntEmbeddingConcat(BuildingBlock):
    """
        convert input features to unique integers over all building blocks
        results are concatenated in the order of the building blocks
    """

    def __init__(self, building_blocks: list[BuildingBlock]):
        super().__init__()
        self._encoding_dict = {idx: {} for idx in range(len(building_blocks))}
        self._dependency_list = building_blocks
        self._size = None

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall):
        """
            takes one feature and assigns it an integer
            the integer is current length of encoding_dict per building_block
            keep 0 free for unknown syscalls
        """
        if self._size is not None:
            return
        for idx, bb in enumerate(self._dependency_list):
            bb_value = bb.get_result(syscall)
            if isinstance(bb_value, (list, tuple)):
                for value in bb_value:
                    if value not in self._encoding_dict[idx]:
                        self._encoding_dict[idx][value] = len(self._encoding_dict[idx]) + 1
            else:
                if bb_value not in self._encoding_dict[idx]:
                    self._encoding_dict[idx][bb_value] = len(self._encoding_dict[idx]) + 1

    def fit(self):
        """ offset encodings and update size """
        if self._size is not None:
            return

        offset = 0
        for idx, encoding_dict in self._encoding_dict.items():
            for key, value in encoding_dict.items():
                encoding_dict[key] = value + offset
            offset += len(encoding_dict)
        self._size = offset + 1  # +1 for unknown

    def _calculate(self, syscall: Syscall):
        """
            transforms given building_block to integer
        """
        result = []
        for idx, bb in enumerate(self._dependency_list):
            bb_value = bb.get_result(syscall)
            if isinstance(bb_value, (list, tuple)):
                result.extend(self._encoding_dict[idx].get(value, 0) for value in bb_value)
            else:
                result.append(self._encoding_dict[idx].get(bb_value, 0))
        return tuple(result)

    def __len__(self):
        return self._size
