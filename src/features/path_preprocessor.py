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

import re

from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class PathPreprocessor(BuildingBlock):
    """ Preprocesses a path by removing special characters and converting to lowercase."""

    def __init__(self, building_block: BuildingBlock):
        super().__init__()
        self._dependency_list = [building_block]

    def _calculate(self, syscall: Syscall):
        """ Tokenizes a path into a list of tokens. """
        result = self._dependency_list[0].get_result(syscall)
        if result is None:
            return None
        return self._preprocess_path(' '.join(result))

    @staticmethod
    def _preprocess_path(path: str):
        """
            remove parenthesis and slashes from a path
            keep only letters and dots
        Examples:
            >>> _preprocess_path("textDB(/usr/bin/ls) ")
            >>> "textDB usr bin ls"
            >>> _preprocess_path("/usr/../../tmp/node ")
            >>> "usr .. .. tmp node"
        Args:
            path: a path string or path like string
        """
        path = path.lower()
        path = re.sub(r"\(/|\)|/", " ", path)
        return " ".join([x for x in path.split(" ") if x.isalpha() or x == ".."])

    def depends_on(self) -> list:
        return self._dependency_list
