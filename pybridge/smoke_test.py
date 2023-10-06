import unittest
import set_path

set_path.append_sys_path()
import bridge

print(dir(bridge))


class TestBridgeModule(unittest.TestCase):

    def test_bridge_game(self):
        bridge_game = bridge.BridgeGame({"is_dealer_vulnerable": "false",
                                         "is_non_dealer_vulnerable": "false"})
        self.assertEqual(bridge_game.num_distinct_actions(), 52 + 38)
        self.assertEqual(bridge_game.max_chance_outcomes(), 52)
        self.assertEqual(bridge_game.max_utility(), 7600)
        self.assertEqual(bridge_game.min_utility(), -7600)

    def test_bridge_state(self):
        bridge_game = bridge.BridgeGame({"is_dealer_vulnerable": "false",
                                         "is_non_dealer_vulnerable": "false"})
        bridge_state = bridge_game.new_initial_state()
        self.assertTrue(True)
        self.assertEqual(bridge_state.is_terminal(), False)
        self.assertEqual(bridge_state.current_phase(), 0)
        # self.assertEqual(bridge_state.contract_index(), 0)
        self.assertEqual(bridge_state.current_player(), -1)
        self.assertEqual(len(bridge_state.history()), 0)
        # print(bridge_state)
        cards = bridge.example_cards[0]
        for card in cards:
            bridge_state.apply_action(card)
        print(bridge_state)
        print(bridge_state.double_dummy_results(dds_order=False))
