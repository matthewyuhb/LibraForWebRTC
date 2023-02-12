import torch
from typing import Any, Dict, Optional, Type, TypeVar, Union, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.ppo.ppo import PPO
from imitation.policies.base import FeedForward32Policy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
import numpy as np

LokiFusionAgentSelf = TypeVar("LokiFusionAgentSelf", bound="LokiFusionAgent")


class LokiFusionAgent(BaseAlgorithm):
    def __init__(self, gcc_model, rl_model):
        self.gcc_model: FeedForward32Policy = torch.load(gcc_model)
        self.rl_model = PPO.load(rl_model)

    def _setup_model(self) -> None:
        return super()._setup_model()

    def learn(self: LokiFusionAgentSelf, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100, tb_log_name: str = "run", eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5, eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True, progress_bar: bool = False) -> LokiFusionAgentSelf:
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, eval_env, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, progress_bar)

    def predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        gcc_obs = np.zeros(10)
        obs = observation.reshape(25)
#         obs =  [  0.,         -11.018043,     1.0416666,    0.35766283,   1.,
#    0. ,         -8.924157 ,    1.0638298,    0.4578084,    1.,
#    0.  ,        -6.7717195 ,   1.0080645  ,  0.5859948,    0.,
#    0.  ,        -3.9711518 ,   0.95238096 ,  0.7500733 ,   0.,
#    0.  ,         0.35515535 ,  0.89285713 ,  0.7650748,    0.,        ]
        # print("obs:{}".format(obs))
        
        for i in range(5):
            gcc_obs[2*i] = obs[i]
            gcc_obs[2*i+1] = obs[5*i+1]
        gcc_tensor, _ = self.gcc_model.obs_to_tensor(gcc_obs)
        
        gcc_distribution = self.gcc_model.get_distribution(gcc_tensor)
        print("gcc_distribution.mode():{}".format(gcc_distribution.mode()))
        
        rl_tensor, _ = self.rl_model.policy.obs_to_tensor(obs[-5:])
        rl_distribution = self.rl_model.policy.get_distribution(
            rl_tensor)  # type: ignore
        gcc_probs = gcc_distribution.distribution.probs.reshape(
            10)  # type: ignore
        print("gcc_probs:{}".format(gcc_probs))
        rl_probs = rl_distribution.distribution.probs.reshape(
            10)  # type: ignore
        print("rl_probs:{}".format(rl_probs))

        gcc_probs_modified = self._process_gcc_distribution(gcc_probs)
        # return gcc_distribution.mode(), None
        return torch.argmax(gcc_probs_modified*rl_probs).reshape(1), None

    def _process_gcc_distribution(self, gcc_distribution: torch.Tensor) -> torch.Tensor:
        _beta = 20
        _gamma = 4
        if torch.argmax(gcc_distribution) <= _gamma:
            gcc_distribution = torch.exp(_beta*gcc_distribution)
        else:
            gcc_distribution = torch.sigmoid(gcc_distribution)
        return gcc_distribution
