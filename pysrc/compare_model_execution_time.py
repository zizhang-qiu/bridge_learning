import os
from typing import Dict
import hydra
import time
import torch
import random


from common_utils import MultiStats
from set_path import append_sys_path
from agent import BridgeA2CModel
from other_models import PBEModel, A2CAgent
import net
from utils import tensor_dict_to_device, tensor_dict_unsqueeze

append_sys_path()
import bridge
import bridgelearn


def get_legal_moves_mask_from_obs(obs: bridge.BridgeObservation) -> torch.Tensor:
    legal_moves = obs.legal_moves()
    legal_moves_mask = torch.zeros(bridge.NUM_CALLS, dtype=torch.float32)
    for move in legal_moves:
        idx = bridge.default_game.get_move_uid(move) - 52
        legal_moves_mask[idx] = 1

    return legal_moves_mask


@hydra.main("conf", "compare_model_execution_time", version_base="1.2")
def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    stats = MultiStats()
    models = {}

    jps_agent: A2CAgent = hydra.utils.instantiate(args.jps_a2c)
    jps_agent.cuda()

    pbe_agent = PBEModel()
    pbe_agent.cuda()
    policy_conf = dict(args.net)
    del policy_conf["_target_"]
    value_conf = dict(args.value_net)
    del value_conf["_target_"]
    belief_conf = dict(args.belief_net)
    del belief_conf["_target_"]
    rl_agent = BridgeA2CModel(policy_conf, value_conf, belief_conf)
    rl_agent.cuda()
    # net = hydra.utils.instantiate(args.net)
    # rl_agent.policy_net = hydra.utils.instantiate(args.net)
    # rl_agent.value_net = hydra.utils.instantiate(args.value_net)
    # rl_agent.belief_net = hydra.utils.instantiate(args.belief_net)
    # rl_agent.policy_net = net
    # print(rl_agent)
    models["jps"] = jps_agent
    models["pbe"] = pbe_agent
    models["rl"] = rl_agent

    num_node = 0
    options = bridgelearn.BridgeEnvOptions()
    options.bidding_phase = True
    options.playing_phase = False
    options.pbe_feature = True
    options.jps_feature = True
    env = bridgelearn.BridgeEnv({}, options)
    # canonical_encoder = bridge.CanonicalEncoder(bridge.default_game)
    # pbe_encoder = bridge.PBEEncoder(bridge.default_game)
    # jps_encoder = bridge.JPSEncoder(bridge.default_game)
    # state = bridge.BridgeState(bridge.default_game)
    # while state.is_chance_node():
    #     state.apply_random_chance()
    while num_node < args.num_nodes:
        # while state.current_phase() == bridge.Phase.PLAY:
        #     legal_moves = state.legal_moves()
        #     move = random.choice(legal_moves)
        #     state.apply_move(move)
            
        # if state.is_terminal():
        #     state = bridge.BridgeState(bridge.default_game)
        #     while state.is_chance_node():
        #         state.apply_random_chance()
        if env.terminated():
            env.reset()

        # For each agent, compute feature and get action.
        # obs = bridge.BridgeObservation(state)
        # pbe_feature = pbe_encoder.encode(obs)
        # jps_feature = jps_encoder.encode(obs)
        # rl_feature = canonical_encoder.encode(obs)[:480]
        obs = env.feature()
        obs = tensor_dict_unsqueeze(obs, 0)
        obs = tensor_dict_to_device(obs, args.device)
        # for k,v  in obs.items():
        #     print(k, v.shape)
        # input()

        # pbe_obs = tensor_dict_to_device(
        #     {"s": torch.tensor(pbe_feature, dtype=torch.float32).unsqueeze(0)}, "cuda"
        # )

        # jps_obs = tensor_dict_to_device(
        #     {
        #         "s": torch.tensor(jps_feature, dtype=torch.float32).unsqueeze(0),
        #         "legal_move": torch.unsqueeze(
        #             torch.tensor(jps_feature, dtype=torch.float32), 0
        #         )[:, -39:],
        #     },
        #     "cuda",
        # )

        # rl_obs = tensor_dict_to_device(
        #     {
        #         "s": torch.tensor(rl_feature, dtype=torch.float32).unsqueeze(0),
        #         "legal_moves": get_legal_moves_mask_from_obs(obs),
        #     },
        #     "cuda",
        # )

        st = time.perf_counter()
        pbe_agent.act(obs)
        ed = time.perf_counter()
        stats.feed("pbe", ed - st)

        st = time.perf_counter()
        jps_agent.act(obs)
        ed = time.perf_counter()
        stats.feed("jps", ed - st)

        st = time.perf_counter()
        rl_agent.act(obs)
        ed = time.perf_counter()
        stats.feed("rl", ed - st)

        num_node += 1
        print(f"\r{num_node}/{args.num_nodes}", end="")

        # choose a random move in state.
        random_move = random.choice(env.ble_state().legal_moves())
        env.step(random_move)

    print()
    print(f"{args.num_nodes} nodes have been played.")
    print("Result:")
    print(f"PBE: {stats.get('pbe').mean()}")
    print(f"JPS: {stats.get('jps').mean()}")
    print(f"RL: {stats.get('rl').mean()}")
    
    stats.save_all(args.save_dir)


if __name__ == "__main__":
    main()
