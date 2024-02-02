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

from math import log

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class LogScale(BuildingBlock):
    """
    Logarithmic scaling of a feature
    """

    def __init__(self, feature: BuildingBlock, base: int, linear_interpolation_value: int):
        """
        Scale a feature logarithmically if it is above a certain value and is an int or float

        Args:
            feature: input feature
            base:  base of the logarithm
            linear_interpolation_value: value below which the feature is not scaled
        """
        super().__init__()
        self._feature = feature
        self._base = base
        self._dependency_list = [feature]
        if linear_interpolation_value is None:
            self._linear_interpolation_value = 0
        else:
            self._linear_interpolation_value = linear_interpolation_value

    def _calculate(self, syscall: Syscall):
        feature_result = self._feature.get_result(syscall)
        if feature_result is None:
            return None
        if not isinstance(feature_result, (int, float)):
            return feature_result
        if feature_result <= self._linear_interpolation_value:
            return feature_result
        return log(feature_result, self._base)

    def depends_on(self) -> list:
        return self._dependency_list
