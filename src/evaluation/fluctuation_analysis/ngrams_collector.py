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


class NgramsCollector(BuildingBlock):
    """
    A fake building block that collects the ngrams from the training, validation and tests set
    Adds methods for ngrams before an exploit starts and after the exploit
    """

    def __init__(self, input_vector: BuildingBlock):
        super().__init__()
        self._input_vector = input_vector
        self._dependency_list = [input_vector]
        self.train_set_length = 0
        self.train_set_counts = {}
        self.test_set_counts = {}
        self.test_set_length = 0
        self.val_set_counts = {}
        self.val_set_length = 0
        self.exploit_set_length = 0
        self.exploit_set_counts = {}
        self.normal_set_length = 0
        self.normal_set_counts = {}
        self.before_exploit_set_length = 0
        self.before_exploit_set_counts = {}
        self.after_exploit_set_length = 0
        self.after_exploit_set_counts = {}
        self.per_rec_before = {}
        self.per_rec_after = {}
        self.per_rec_normal = {}
        self.current_rec = None
        self.syscall_dict = {}

    def train_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.train_set_length += 1
            self.train_set_counts[input_vector] = self.train_set_counts.get(input_vector, 0) + 1

    def val_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.val_set_length += 1
            self.val_set_counts[input_vector] = self.val_set_counts.get(input_vector, 0) + 1

    def test_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.test_set_length += 1
            self.test_set_counts[input_vector] = self.test_set_counts.get(input_vector, 0) + 1

    def exploit_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.exploit_set_length += 1
            self.exploit_set_counts[input_vector] = self.exploit_set_counts.get(input_vector, 0) + 1

    def normal_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.normal_set_length += 1
            self.normal_set_counts[input_vector] = self.normal_set_counts.get(input_vector, 0) + 1
            self.per_rec_normal[self.current_rec] = self.per_rec_normal.get(self.current_rec, {})
            self.per_rec_normal[self.current_rec][input_vector] = \
                self.per_rec_normal[self.current_rec].get(input_vector, 0) + 1

    def before_exploit_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.before_exploit_set_length += 1
            self.before_exploit_set_counts[input_vector] = \
                self.before_exploit_set_counts.get(input_vector, 0) + 1
            self.per_rec_before[self.current_rec] = self.per_rec_before.get(self.current_rec, {})
            self.per_rec_before[self.current_rec][input_vector] = \
                self.per_rec_before[self.current_rec].get(input_vector, 0) + 1

    def after_exploit_on(self, syscall):
        input_vector: tuple = self._input_vector.get_result(syscall)
        if input_vector is not None:
            self.after_exploit_set_length += 1
            self.after_exploit_set_counts[input_vector] = self.after_exploit_set_counts.get(input_vector, 0) + 1
            self.per_rec_after[self.current_rec] = self.per_rec_after.get(self.current_rec, {})
            self.per_rec_after[self.current_rec][input_vector] = \
                self.per_rec_after[self.current_rec].get(input_vector, 0) + 1

    def fit(self):
        pass

    def depends_on(self) -> list:
        return self._dependency_list

    def recording_norm(self, recording_name: str):
        self.current_rec = recording_name
        self.per_rec_normal[recording_name] = {}

    def recording_exploit(self, recording_name: str):
        self.current_rec = recording_name
        self.per_rec_before[recording_name] = {}
        self.per_rec_after[recording_name] = {}