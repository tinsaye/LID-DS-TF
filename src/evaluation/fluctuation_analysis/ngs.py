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
