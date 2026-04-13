"""Tests for expanded battle/PC state in GameState."""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_state import (
    GameState,
    IN_BATTLE,
    OPPONENT_ACTIVE_SPECIES,
    OPPONENT_ACTIVE_LEVEL,
    OPPONENT_ACTIVE_HP,
    OPPONENT_ACTIVE_MAX_HP,
    OPPONENT_ACTIVE_TYPE1,
    OPPONENT_ACTIVE_TYPE2,
    PARTY_TYPE1,
    PARTY_TYPE2,
    PARTY_MOVES,
    PARTY_MOVE_PP,
    NUM_POKEMON_IN_BOX,
)


class MockMemory:
    def __init__(self):
        self._data = {}
    def __getitem__(self, addr):
        return self._data.get(addr, 0)
    def __setitem__(self, addr, value):
        self._data[addr] = value


class MockPyBoy:
    def __init__(self):
        self.memory = MockMemory()


@pytest.fixture
def gs():
    pyboy = MockPyBoy()
    return GameState(pyboy), pyboy


class TestBattleState:
    def test_battle_type(self, gs):
        state, pyboy = gs
        assert state.battle_type == 0
        pyboy.memory[IN_BATTLE] = 2
        assert state.battle_type == 2

    def test_opponent_species(self, gs):
        state, pyboy = gs
        pyboy.memory[OPPONENT_ACTIVE_SPECIES] = 0x99
        assert state.opponent_species == 0x99

    def test_opponent_level(self, gs):
        state, pyboy = gs
        pyboy.memory[OPPONENT_ACTIVE_LEVEL] = 25
        assert state.opponent_level == 25

    def test_opponent_hp_fraction(self, gs):
        state, pyboy = gs
        pyboy.memory[OPPONENT_ACTIVE_HP] = 0
        pyboy.memory[OPPONENT_ACTIVE_HP + 1] = 30
        pyboy.memory[OPPONENT_ACTIVE_MAX_HP] = 0
        pyboy.memory[OPPONENT_ACTIVE_MAX_HP + 1] = 100
        assert state.opponent_hp_fraction == 0.3

    def test_opponent_hp_fraction_zero_max(self, gs):
        state, _ = gs
        assert state.opponent_hp_fraction == 1.0

    def test_opponent_types(self, gs):
        state, pyboy = gs
        pyboy.memory[OPPONENT_ACTIVE_TYPE1] = 0x14  # Fire
        pyboy.memory[OPPONENT_ACTIVE_TYPE2] = 0x02  # Flying
        assert state.opponent_types == (0x14, 0x02)
        assert state.opponent_type_names == ("Fire", "Flying")


class TestPartyDetails:
    def test_party_types(self, gs):
        state, pyboy = gs
        pyboy.memory[PARTY_TYPE1[0]] = 0x15  # Water
        pyboy.memory[PARTY_TYPE2[0]] = 0x19  # Ice
        types = state.party_types
        assert types[0] == (0x15, 0x19)

    def test_lead_pokemon_types(self, gs):
        state, pyboy = gs
        pyboy.memory[PARTY_TYPE1[0]] = 0x16  # Grass
        pyboy.memory[PARTY_TYPE2[0]] = 0x03  # Poison
        assert state.lead_pokemon_types == (0x16, 0x03)

    def test_party_moves(self, gs):
        state, pyboy = gs
        for i, addr in enumerate(PARTY_MOVES[0]):
            pyboy.memory[addr] = 10 + i
        moves = state.party_moves
        assert moves[0] == [10, 11, 12, 13]

    def test_lead_moves(self, gs):
        state, pyboy = gs
        for i, addr in enumerate(PARTY_MOVES[0]):
            pyboy.memory[addr] = 33 + i
        assert state.lead_moves == [33, 34, 35, 36]

    def test_lead_pp(self, gs):
        state, pyboy = gs
        for i, addr in enumerate(PARTY_MOVE_PP[0]):
            pyboy.memory[addr] = 20 + i
        assert state.lead_pp == [20, 21, 22, 23]


class TestPCState:
    def test_box_not_full(self, gs):
        state, pyboy = gs
        pyboy.memory[NUM_POKEMON_IN_BOX] = 5
        assert state.pokemon_in_current_box == 5
        assert state.is_box_full is False

    def test_box_full(self, gs):
        state, pyboy = gs
        pyboy.memory[NUM_POKEMON_IN_BOX] = 20
        assert state.is_box_full is True

    def test_box_over_full(self, gs):
        state, pyboy = gs
        pyboy.memory[NUM_POKEMON_IN_BOX] = 25
        assert state.is_box_full is True
