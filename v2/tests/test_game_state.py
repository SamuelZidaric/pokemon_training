"""Tests for GameState — uses a mock memory to avoid needing a real ROM."""

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_state import (
    GameState,
    Position,
    PLAYER_X,
    PLAYER_Y,
    MAP_NUMBER,
    PARTY_SIZE,
    PARTY_LEVELS,
    PARTY_HP,
    PARTY_MAX_HP,
    BADGES,
    IN_BATTLE,
    EVENT_FLAGS_START,
    EVENT_FLAGS_END,
)


class MockMemory:
    """Dict-backed memory that behaves like pyboy.memory."""

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


class TestPosition:
    def test_str(self):
        p = Position(x=5, y=10, map_id=3)
        assert str(p) == "x:5 y:10 m:3"

    def test_frozen(self):
        p = Position(x=1, y=2, map_id=3)
        with pytest.raises(AttributeError):
            p.x = 10


class TestGameState:
    def test_position(self, gs):
        state, pyboy = gs
        pyboy.memory[PLAYER_X] = 42
        pyboy.memory[PLAYER_Y] = 13
        pyboy.memory[MAP_NUMBER] = 7
        pos = state.position
        assert pos.x == 42
        assert pos.y == 13
        assert pos.map_id == 7

    def test_in_battle(self, gs):
        state, pyboy = gs
        assert state.in_battle is False
        pyboy.memory[IN_BATTLE] = 1
        assert state.in_battle is True

    def test_party_size(self, gs):
        state, pyboy = gs
        pyboy.memory[PARTY_SIZE] = 3
        assert state.party_size == 3

    def test_party_levels(self, gs):
        state, pyboy = gs
        for i, addr in enumerate(PARTY_LEVELS):
            pyboy.memory[addr] = (i + 1) * 10
        assert state.party_levels == [10, 20, 30, 40, 50, 60]
        assert state.levels_sum == 210

    def test_hp_fraction(self, gs):
        state, pyboy = gs
        # Set first pokemon: HP = 50 (0x00, 0x32), Max HP = 100 (0x00, 0x64)
        pyboy.memory[PARTY_HP[0]] = 0
        pyboy.memory[PARTY_HP[0] + 1] = 50
        pyboy.memory[PARTY_MAX_HP[0]] = 0
        pyboy.memory[PARTY_MAX_HP[0] + 1] = 100
        assert state.hp_fraction == 0.5

    def test_hp_fraction_zero_max(self, gs):
        state, _ = gs
        # All zeros — should return 1.0, not crash
        assert state.hp_fraction == 1.0

    def test_badge_count(self, gs):
        state, pyboy = gs
        pyboy.memory[BADGES] = 0b00001111
        assert state.badge_count == 4

    def test_badges_array(self, gs):
        state, pyboy = gs
        pyboy.memory[BADGES] = 0b10000001
        arr = state.badges_array
        assert arr[0] == 1  # MSB
        assert arr[7] == 1  # LSB
        assert sum(arr) == 2

    def test_read_bit(self, gs):
        state, pyboy = gs
        pyboy.memory[0xAAAA] = 0b00000100
        assert state.read_bit(0xAAAA, 2) is True
        assert state.read_bit(0xAAAA, 0) is False

    def test_read_word(self, gs):
        state, pyboy = gs
        pyboy.memory[0x1000] = 0x01
        pyboy.memory[0x1001] = 0x00
        assert state.read_word(0x1000) == 256

    def test_bit_count(self):
        assert GameState.bit_count(0b11110000) == 4
        assert GameState.bit_count(0) == 0
        assert GameState.bit_count(0xFF) == 8
