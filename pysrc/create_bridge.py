from typing import Dict, List

from agent import BridgeA2CModel
from wbridge5_client import WBridge5Client
from set_path import append_sys_path

append_sys_path()
import bridge
import torch
import pyrela
import bridgeplay

from utils import load_net_conf_and_state_dict

from bluechip_bot import BlueChipBridgeBot


def create_params(is_dealer_vulnerable: bool = False, is_non_dealer_vulnerable: bool = False,
                  dealer: int = 0, seed: int = 0) -> Dict[str, str]:
    params = {
        "is_dealer_vulnerable": str(is_dealer_vulnerable),
        "is_non_dealer_vulnerable": str(is_non_dealer_vulnerable),
        "dealer": str(dealer),
        "seed": str(seed)
    }
    return params


def create_bridge_game(params=None) -> bridge.BridgeGame:
    if params is None:
        params = create_params()
    return bridge.BridgeGame(params)


class BotFactory:
    # For belief-based bot.
    belief_model_dir: str
    belief_model_name: str
    policy_model_dir: str
    policy_model_name: str
    device: str

    # For pimc
    seed: int
    num_worlds: int

    # For alphamu
    num_max_moves: int
    early_cut: bool
    root_cut: bool
    rollout_result : int

    # For BBA
    convention_file: str
    bidding_systems: List[int]

    # Others
    fill_with_uniform_sample: bool
    num_max_sample: int
    verbose: bool

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_attr_from_kwargs_if_exists(self, attr_name: str, kwargs:Dict):
        if attr_name in kwargs.keys():
            return kwargs[attr_name]
        return getattr(self, attr_name)

    def create_bot(self, bot_type: str, **kwargs) -> bridgeplay.PlayBot:
        if bot_type == "dds":
            return bridgeplay.DDSBot()

        if bot_type.lower() == "pimc":
            resampler = bridgeplay.UniformResampler(self.seed)
            pimc_config = bridgeplay.PIMCConfig()
            pimc_config.num_worlds = self.get_attr_from_kwargs_if_exists("num_worlds", kwargs)
            pimc_config.search_with_one_legal_move = False
            return bridgeplay.PIMCBot(resampler, pimc_config)

        if bot_type.lower() == "alpha_mu":
            assert "player_id" in kwargs.keys()
            resampler = bridgeplay.UniformResampler(self.seed)
            alpha_mu_config = bridgeplay.AlphaMuConfig()
            alpha_mu_config.num_worlds = self.get_attr_from_kwargs_if_exists(
                "num_worlds", kwargs
            )
            alpha_mu_config.num_max_moves = self.get_attr_from_kwargs_if_exists(
                "num_max_moves", kwargs
            )
            alpha_mu_config.root_cut = self.get_attr_from_kwargs_if_exists(
                "root_cut", kwargs
            )
            alpha_mu_config.early_cut = self.get_attr_from_kwargs_if_exists(
                "early_cut", kwargs
            )
            alpha_mu_config.search_with_one_legal_move = False
            alpha_mu_config.rollout_result = bridgeplay.RolloutResult(
                self.get_attr_from_kwargs_if_exists("rollout_result", kwargs)
            )
            return bridgeplay.AlphaMuBot(resampler, alpha_mu_config, kwargs["player_id"])

        if "game" in kwargs.keys():
            game = kwargs['game']
        else:
            game = bridge.default_game

        dds_evaluator = bridgeplay.DDSEvaluator()

        if bot_type == "nn_belief_opening" or bot_type=="NNB-OL":
            if "torch_actor" in kwargs.keys():
                torch_actor = kwargs["torch_actor"]
            else:
                policy_conf, policy_state_dict = load_net_conf_and_state_dict(self.policy_model_dir,
                                                                              self.policy_model_name)
                belief_conf, belief_state_dict = load_net_conf_and_state_dict(self.belief_model_dir,
                                                                              self.belief_model_name)

                agent = BridgeA2CModel(
                    policy_conf=policy_conf,
                    value_conf=dict(
                        hidden_size=2048,
                        num_hidden_layers=6,
                        use_layer_norm=True,
                        activation_function="gelu",
                        output_size=1
                    ),
                    belief_conf=belief_conf
                )
                agent.policy_net.load_state_dict(policy_state_dict)
                agent.belief_net.load_state_dict(belief_state_dict)
                agent.to(self.device)
                print("Network loaded.")

                batch_runner = pyrela.BatchRunner(agent, self.device, 100, ["get_policy", "get_belief"])
                batch_runner.start()

                torch_actor = bridgeplay.TorchActor(batch_runner)

            cfg = bridgeplay.BeliefBasedOpeningLeadBotConfig()
            cfg.num_worlds = self.get_attr_from_kwargs_if_exists("num_worlds", kwargs)
            cfg.num_max_sample = self.get_attr_from_kwargs_if_exists(
                "num_max_sample", kwargs
            )
            cfg.rollout_result = bridgeplay.RolloutResult(
                self.get_attr_from_kwargs_if_exists("rollout_result", kwargs)
            )
            cfg.fill_with_uniform_sample = self.get_attr_from_kwargs_if_exists(
                "fill_with_uniform_sample", kwargs
            )
            cfg.verbose = self.get_attr_from_kwargs_if_exists("verbose", kwargs)
            nn_belief_opening_bot = bridgeplay.NNBeliefOpeningLeadBot(torch_actor, game, self.seed,
                                                                      dds_evaluator, cfg)
            return nn_belief_opening_bot

        if bot_type == "bba":
            from bba import load_conventions
            conventions = load_conventions(self.convention_file)
            from bba_bot import BBABot
            assert "player_id" in kwargs.keys()
            player_id = kwargs["player_id"]

            return BBABot(player_id, game, self.bidding_systems, conventions)

        if bot_type == "rule_based_opening" or bot_type=="RBB-OL":
            from rule_based_bot import RuleBasedBot
            from bba import load_conventions
            conventions = load_conventions(self.convention_file)
            cfg = bridgeplay.BeliefBasedOpeningLeadBotConfig()
            cfg.num_worlds = self.get_attr_from_kwargs_if_exists("num_worlds", kwargs)
            cfg.num_max_sample = self.get_attr_from_kwargs_if_exists("num_max_sample", kwargs)
            cfg.rollout_result = bridgeplay.RolloutResult(self.get_attr_from_kwargs_if_exists("rollout_result", kwargs))
            cfg.fill_with_uniform_sample = self.get_attr_from_kwargs_if_exists(
                "fill_with_uniform_sample", kwargs
            )
            cfg.verbose = self.verbose
            return RuleBasedBot(game, self.bidding_systems, conventions, dds_evaluator, cfg)

        if bot_type == "bluechip":
            assert "player_id" in kwargs.keys()
            player_id = kwargs["player_id"]
            assert "cmd_line" in kwargs.keys()
            cmd_line = kwargs["cmd_line"]

            timeout_secs = kwargs.get("timeout_secs", 60)

            def controller_factory():
                client = WBridge5Client(cmd_line, timeout_secs)
                client.start()
                return client

            return BlueChipBridgeBot(game, player_id, controller_factory)
        
        if bot_type == "wbridge5_trajectory":
            assert "trajectories" in kwargs.keys()
            return bridgeplay.WBridge5TrajectoryBot(kwargs["trajectories"], game)

        raise ValueError(f"bot_type {bot_type} is not supported.")
