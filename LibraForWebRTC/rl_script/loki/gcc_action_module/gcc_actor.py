import imitation.policies
from imitation.data import rollout
import numpy as np
from enum import Enum

_THRESHOLD = 12.5
_K_DOWN = 0.039
_K_UP = 0.0187


class GCCSTATE(Enum):
    BW_NORMAL = 0
    BW_UNDERUSING = 1
    BW_OVERUSING = 2


class GCCActor:
    def __init__(self) -> None:
        self.threshold = _THRESHOLD
        self.gcc_state = GCCSTATE.BW_NORMAL
        # self.aimd_state = AIMDState.AIMD_INCREASE
        # 是否初始化过
        self.inited = False

    def get_state(self, trendline) -> int:

        # 判断当前网络状态
        self._overuse_detect(trendline)

        return self.gcc_state.value

    def _overuse_detect(self, trendline):
        '''
        利用Trendline来判断网络是否过载，并更新过载阈值
        '''
        if np.fabs(trendline) < self.threshold:
            self.gcc_state = GCCSTATE.BW_NORMAL
        elif trendline > self.threshold:
            self.gcc_state = GCCSTATE.BW_OVERUSING
        else:
            self.gcc_state = GCCSTATE.BW_UNDERUSING

        self._overuse_update_threshold(trendline)

    def _overuse_update_threshold(self, trendline):
        '''
        更新过载阈值
        '''
        k = _K_DOWN if np.fabs(trendline) < self.threshold else _K_UP
        self.threshold += k * (np.fabs(trendline) - self.threshold)
        self.threshold = np.clip(self.threshold, 6, 600)
