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

class Ngs:
    """ Ngram sets """
    def __init__(self):
        self.train_set = []
        self.val_set = []
        self.exploit_set = []
        self.normal_set = []
        self.before_exploit_set = []
        self.after_exploit_set = []
        self.per_rec_before = {}
        self.per_rec_after = {}
        self.per_rec_normal = {}
        self.val_exc_train = []
        self.exploit_exc_train = []
        self.normal_exc_train = []
        self.exploit_exc_val = []
        self.normal_exc_val = []
        self.exploit_exc_train_val = []
        self.normal_exc_train_val = []
        self.before_exploit_exc_train_val = []
        self.after_exploit_exc_train_val = []
        self.before_exploit_exc_train = []
        self.after_exploit_exc_train = []
        self.train_set_split = []
        self.val_set_split = []
        self.all_set = []
        self.true_all_len = 0
        self.train_set_len = 0
        self.val_set_len = 0
        self.exploit_set_len = 0
        self.normal_set_len = 0
        self.before_exploit_set_len = 0
        self.after_exploit_set_len = 0
