import hydra
from set_path import append_sys_path
append_sys_path()

import bridge
from net import MLP

@hydra.main("conf", "evaluate_belief_net", version_base="1.2")
def main(args):
    policy_net:MLP = hydra.utils.instantiate(args.net)
    belief_net: MLP = hydra.utils.instantiate(args.belief_net)
    print(policy_net)
    print(belief_net)
    
if __name__ == "__main__":
    main()