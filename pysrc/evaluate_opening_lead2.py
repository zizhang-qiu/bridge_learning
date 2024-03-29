"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: evaluate_opening_lead2.py
@time: 2024/1/22 9:36
"""

import argparse
import os
from typing import List
import time
import numpy as np
import multiprocessing as mp


import set_path
import hydra
from omegaconf import OmegaConf, DictConfig

set_path.append_sys_path()
import common_utils
import bridge
import rela
import bridgeplay
from create_bridge import BotFactory
from utils import extract_not_passed_out_trajectories


def construct_deal_and_bidding_state(
    trajectory: List[int], game: bridge.BridgeGame = bridge.default_game
) -> bridge.BridgeState:
    assert len(trajectory) > game.min_game_length()
    state = bridge.BridgeState(game)
    idx = 0
    while not state.current_phase() == bridge.Phase.PLAY:
        uid = trajectory[idx]
        if state.is_chance_node():
            move = game.get_chance_outcome(uid)
        else:
            move = game.get_move(uid)
        state.apply_move(move)
        idx += 1
    return state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default=r"D:\Projects\bridge_research\expert"
    )
    parser.add_argument("--policy_model_dir", type=str, default="sl/exp6")
    parser.add_argument("--policy_model_name", type=str, default="model0.pthw")
    parser.add_argument("--belief_model_dir", type=str, default="belief_sl/exp3")
    parser.add_argument("--belief_model_name", type=str, default="model2.pthw")

    parser.add_argument("--num_worlds", type=int, default=160)
    parser.add_argument("--num_max_sample", type=int, default=1600)
    parser.add_argument("--fill_with_uniform_sample", type=int, default=1)

    parser.add_argument("--num_threads", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class Worker(mp.Process):
    def __init__(
        self,
        flags,
        trajectories: List[List[int]],
        q: mp.SimpleQueue,
        aq: mp.SimpleQueue,
        process_idx: int = 0,
    ):
        super().__init__()
        self.args = flags
        self.trajectories = trajectories
        self.aq = aq
        self.q = q
        self.process_idx = process_idx

    def run(self):
        dds_evaluator = bridgeplay.DDSEvaluator()

        # Create agent
        # policy_conf, policy_state_dict = load_net_conf_and_state_dict(self.args.policy_model_dir,
        #                                                               self.args.policy_model_name)
        # belief_conf, belief_state_dict = load_net_conf_and_state_dict(self.args.belief_model_dir,
        #                                                               self.args.belief_model_name)
        #
        # agent = BridgeA2CModel(
        #     policy_conf=policy_conf,
        #     value_conf=dict(
        #         hidden_size=2048,
        #         num_hidden_layers=6,
        #         use_layer_norm=True,
        #         activation_function="gelu",
        #         output_size=1
        #     ),
        #     belief_conf=belief_conf
        # )
        # agent.policy_net.load_state_dict(policy_state_dict)
        # agent.belief_net.load_state_dict(belief_state_dict)
        # agent.to(self.args.device)
        # print("Network loaded.")
        #
        # batch_runner = rela.BatchRunner(agent, self.args.device, 100, ["get_policy", "get_belief"])
        # batch_runner.start()

        # cfg = bridgeplay.BeliefBasedOpeningLeadBotConfig()
        # cfg.num_worlds = self.args.num_worlds
        # cfg.num_max_sample = self.args.num_max_sample
        # cfg.fill_with_uniform_sample = bool(self.args.fill_with_uniform_sample)
        # cfg.verbose = False
        # cfg.rollout_result = bridgeplay.RolloutResult.NUM_FUTURE_TRICKS

        # torch_actor = bridgeplay.TorchActor(batch_runner)
        # bot = bridgeplay.TorchOpeningLeadBot(torch_actor, bridge.default_game, 1, dds_evaluator, cfg)

        # pimc_cfg = bridgeplay.PIMCConfig()
        # pimc_cfg.num_worlds = self.args.num_worlds
        # pimc_cfg.search_with_one_legal_move = False
        # resampler = bridgeplay.UniformResampler(1)
        # # bot = bridgeplay.PIMCBot(resampler, pimc_cfg)
        # # bot = bridgeplay.WBridge5TrajectoryBot(self.trajectories, bridge.default_game)
        # conventions_list = bba_bot.load_conventions("conf/bidding_system/WBridge5-SAYC.bbsa")

        # bot = RuleBasedBot(bridge.default_game,
        #                    [1, 1], conventions_list,
        #                    dds_evaluator, cfg)  # 779 798 798 804
        bot_factory: BotFactory = hydra.utils.instantiate(self.args.bot_factory)
        opening_lead_bots = [bot_factory.create_bot(self.args.bot_name, trajectories=self.trajectories) for i in range(bridge.NUM_PLAYERS)]
        num_match = 0
        num_actual_match = 0
        num_actual_deals = 0
        logger = common_utils.Logger(
            os.path.join(self.args.save_dir, f"logs_{self.process_idx}.txt"),
            verbose=False,
            auto_line_feed=True,
        )
        execution_times = []
        
        for j, trajectory in enumerate(self.trajectories):
            for bot in opening_lead_bots:
                bot.restart()
            state = construct_deal_and_bidding_state(trajectory)
            # print(state)
            assert not state.is_terminal()
            # Get dds moves.
            dds_moves = dds_evaluator.dds_moves(state)

            # Get bot's move
            st =time.perf_counter()
            bot_move = opening_lead_bots[state.current_player()].step(state)
            ed = time.perf_counter()
            execution_times.append(ed - st)
            msg = f"Deal {j}, DDS moves:\n{dds_moves}\nBot move:{bot_move}"
            logger.write(msg)
            
            if not len(dds_moves) == bridge.NUM_CARDS_PER_HAND:
                num_actual_deals += 1

            if bot_move in dds_moves:
                num_match += 1
                # self.q.put(1)
                if not len(dds_moves) == bridge.NUM_CARDS_PER_HAND:
                    # self.aq.put(1)
                    num_actual_match += 1
            # else:
            #     self.q.put(0)
            #     self.aq.put(0)

            print(
                f"Process {self.process_idx}, ddolar: {num_match}/{j + 1}, addolar: {num_actual_match}/{num_actual_deals}, total: {len(self.trajectories)}"
            )
            
            np.save(os.path.join(self.args.save_dir, f"execution_times_{self.process_idx}.npy"), np.array(execution_times))


@hydra.main("conf", "opening_lead", version_base="1.2")
def main(args: DictConfig):
    # Load dataset
    with open(os.path.join(args.dataset_dir, "test.txt"), "r") as f:
        lines = f.readlines()
    test_dataset = []

    for i in range(len(lines)):
        line = lines[i].split(" ")
        test_dataset.append([int(x) for x in line])

    test_dataset = extract_not_passed_out_trajectories(test_dataset)[:args.num_deals]
    datasets = common_utils.allocate_list_uniformly(test_dataset, args.num_processes)

    queue = mp.SimpleQueue()
    actual_queue = mp.SimpleQueue()

    workers = []
    for i in range(args.num_processes):
        worker = Worker(args, datasets[i], queue, actual_queue, i)
        workers.append(worker)

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    results = []
    while not queue.empty():
        item = queue.get()
        results.append(item)

    results2 = []
    while not actual_queue.empty():
        item = actual_queue.get()
        results2.append(item)

    print(f"DDOLAR: {sum(results)}/{len(results)}")
    print(f"ADDOLAR: {sum(results2)}/{len(results2)}")
    
    exec_times = []
    for i in range(args.num_processes):
        exec_times.append(np.load(os.path.join(args.save_dir, f"execution_times_{i}.npy")))
    exec_times = np.concatenate(exec_times)

    print(f"{args.bot_name} exec time: {common_utils.get_avg_and_sem(exec_times)}")


if __name__ == "__main__":
    main()
