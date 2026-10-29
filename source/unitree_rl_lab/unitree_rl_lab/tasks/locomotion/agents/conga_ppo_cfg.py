from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class CongaBasicPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2_conga_basic"
    max_iterations = 1000
    # Match stock sample budget; network layout matches every conga stage.
    num_steps_per_env = 24


@configclass
class CongaMarchBalancedPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_march_balanced"
    save_interval = 25


@configclass
class CongaMarchPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_march"
    save_interval = 25


@configclass
class CongaPausePPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_pause"
    save_interval = 10


@configclass
class CongaHistoryPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_history"
    save_interval = 10


@configclass
class CongaEarlyPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_early"
    save_interval = 50


@configclass
class CongaCadenceStartPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_cadence_start"


@configclass
class CongaCadencePairedPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = "go2_conga_cadence_paired"


@configclass
class CongaPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2_conga"
    # More than one beat per rollout; the reference is observable, so a recurrent
    # actor is unnecessary. All stages share this actor/critic layout.
    num_steps_per_env = 48


@configclass
class CongaWalkValidationPPORunnerCfg(CongaPPORunnerCfg):
    experiment_name = "go2_conga_walk_validation"
    max_iterations = 200
    save_interval = 50


@configclass
class CongaDurationPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = 'go2_conga_duration'
    save_interval = 25


@configclass
class CongaDurationStrongPPORunnerCfg(CongaDurationPPORunnerCfg):
    experiment_name = 'go2_conga_duration_strong'


@configclass
class CongaDurationAveragePPORunnerCfg(CongaDurationPPORunnerCfg):
    experiment_name = 'go2_conga_duration_average'


@configclass
class CongaKickTrainPPORunnerCfg(CongaBasicPPORunnerCfg):
    experiment_name = 'go2_conga_kick_intro'
    save_interval = 25
