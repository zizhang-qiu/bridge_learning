from typing import Callable, Optional
from wbridge5_client import Controller
from bluechip_bridge import _SEATS, _connect, _new_deal, _hand_string, _expect_regex, _READY_FOR_OTHER, \
    _OTHER_PLAYER_ACTION, _action_to_string, _DUMMY_CARDS, _PLAYER_TO_LEAD, _PLAYER_ACTION, _ACTION_PASS, _ACTION_DBL, \
    _ACTION_RDBL, _bid_to_action, _play_to_action

import set_path

set_path.append_sys_path()

import bridge
import bridgeplay


class BlueChipBridgeBot(bridgeplay.PlayBot):

    def __init__(self, game: bridge.BridgeGame, player_id: int, controller_factory: Callable[[], Controller]):
        super().__init__()
        self._game = game
        self._player_id = player_id
        self._controller_factory = controller_factory
        self._controller: Optional[Controller] = None
        self._seat = _SEATS[player_id]
        self.dummy: Optional[int] = None
        self.is_play_phase = False
        self.cards_played = 0
        self._num_actions = bridge.NUM_CARDS
        self._state = bridge.BridgeState(self._game)
        self._board = 0

    def restart(self):
        """Indicates that we are starting a new episode."""
        # If we already have a fresh state, there is nothing to do.
        if not self._state.history():
            return
        self._num_actions = bridge.NUM_CARDS
        self.dummy = None
        self.is_play_phase = False
        self.cards_played = 0
        if not self._state.is_terminal():
            self._controller.terminate()
            self._controller = None
        self._state = bridge.BridgeState(self._game)

    def inform_state(self, state: bridge.BridgeState):
        # Connect if we need to.
        if self._controller is None:
            self._controller = self._controller_factory()
            _connect(controller=self._controller, seat=self._seat)

        full_history = state.uid_history()
        known_history = self._state.uid_history()

        if full_history[:len(known_history)] != known_history:
            raise ValueError(
                "Supplied state is inconsistent with bot's internal state\n"
                f"Supplied state:\n{state}\n"
                f"Internal state:\n{self._state}\n")

        for uid in full_history[len(known_history):]:
            if self._state.current_phase() == bridge.Phase.DEAL:
                move = self._game.get_chance_outcome(uid)
            else:
                move = self._game.get_move(uid)
            self._state.apply_move(move)
            if not self._state.is_chance_node():
                self._update_for_state()

    def _update_for_state(self):
        """Called for all non-chance nodes, whether or not we have to act."""
        uid_history = self._state.uid_history()
        self.is_play_phase = (self._state.current_phase() == bridge.Phase.PLAY)
        self.cards_played = self._state.num_cards_played()

        # If this is the first time we've seen the deal, send our hand.
        if len(uid_history) == 52:
            self._board += 1
            _new_deal(controller=self._controller,
                      seat=self._seat,
                      hand=_hand_string(uid_history[self._player_id:52:4]),
                      board=str(self._board))

        # Send actions since last `step` call.
        for other_player_action in uid_history[self._num_actions:]:
            other = _expect_regex(controller=self._controller,
                                  regex=_READY_FOR_OTHER.format(seat=self._seat))
            other_player = other["other"]
            if other_player == "Dummy":
                other_player = _SEATS[self.dummy]
            self._controller.send_line(
                line=_OTHER_PLAYER_ACTION.format(
                    player=other_player,
                    action=_action_to_string(other_player_action)))
        self._num_actions = len(uid_history)

        # If the opening lead has just been made, give the dummy.
        if self.is_play_phase and self.cards_played == 1:
            self.dummy = self._state.current_player() ^ 2
            if self._player_id != self.dummy:
                other = _expect_regex(self._controller,
                                      _READY_FOR_OTHER.format(seat=self._seat))
                dummy_cards = _hand_string(uid_history[self.dummy:52:4])
                self._controller.send_line(_DUMMY_CARDS.format(dummy_cards))

        # If the episode is terminal, send (fake) timing info.
        if self._state.is_terminal():
            self._controller.send_line(
                "Timing - N/S : this board  [1:15],  total  [0:11:23].  "
                "E/W : this board  [1:18],  total  [0:10:23]"
            )
            self.dummy = None
            self.is_play_phase = False
            self.cards_played = 0

    def step(self, state):
        """Returns an action for the given state."""
        # Bring the external bot up-to-date.
        self.inform_state(state)

        # If we're on a new trick, tell the bot it is its turn.
        if self.is_play_phase and self.cards_played % 4 == 0:
            self._controller.send_line(_PLAYER_TO_LEAD.format(seat=self._seat))

        # Get our action from the bot.
        our_action = _expect_regex(self._controller, _PLAYER_ACTION)
        self._num_actions += 1
        if our_action["pass"]:
            return _ACTION_PASS
        elif our_action["dbl"]:
            return _ACTION_DBL
        elif our_action["rdbl"]:
            return _ACTION_RDBL
        elif our_action["bid"]:
            return _bid_to_action(our_action["bid"])
        elif our_action["play"]:
            return _play_to_action(our_action["play"])

    def terminate(self):
        self._controller.terminate()
        self._controller = None
