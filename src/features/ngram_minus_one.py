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
from algorithms.features.impl.ngram import Ngram
from dataloader.syscall import Syscall


class NgramMinusOne(BuildingBlock):
    """

    calculate ngram form a stream of system call features
    remove last syscall feature in collect_features
    (Can be later used to fill in syscall int for prediction)

    """

    def __init__(self, ngram: Ngram, element_size: int = None):
        super().__init__()
        self._dependency_list = []
        self._dependency_list.append(ngram)
        self._ngram = ngram
        self._element_size = element_size

    def depends_on(self):
        return self._dependency_list

    def _calculate(self, syscall: Syscall):
        """
        Returns:
            nothing if no ngram exists
            k (int),v (list): key is ID of this class, ngram_value as tuple
        """
        ngram_value = self._ngram.get_result(syscall)
        if ngram_value is not None:
            if self._element_size is None:
                self._element_size = len(ngram_value) // self._ngram._ngram_length - 1
                self._BuildingBlock__config["element_size"] = self._element_size
            return ngram_value[:-self._element_size]
        else:
            return None
