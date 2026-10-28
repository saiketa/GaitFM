from exp.exp_gaitfm_joint_pretrain import Exp_GaitFM_JointPretrain
from models import GaitFMPretrain


class Exp_GaitFMPretrain(Exp_GaitFM_JointPretrain):

    def _build_model(self):
        return GaitFMPretrain.Model(self.args).float()
