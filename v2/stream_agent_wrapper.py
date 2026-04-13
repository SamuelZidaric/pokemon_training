import asyncio
import logging
import json
from typing import Any, Optional

import gymnasium as gym

from game_state import PLAYER_X, PLAYER_Y, MAP_NUMBER

logger = logging.getLogger(__name__)


class StreamWrapper(gym.Wrapper):
    """Wraps a RedGymEnv to broadcast agent coordinates via WebSocket.

    Includes automatic reconnection with exponential backoff so a
    transient server outage doesn't crash training.
    """

    MAX_RECONNECT_DELAY: float = 60.0
    INITIAL_RECONNECT_DELAY: float = 1.0

    def __init__(
        self,
        env: gym.Env,
        stream_metadata: Optional[dict[str, Any]] = None,
        ws_address: str = "wss://transdimensional.xyz/broadcast",
        upload_interval: int = 300,
    ) -> None:
        super().__init__(env)
        self.ws_address = ws_address
        self.stream_metadata = stream_metadata or {}
        self.upload_interval = upload_interval

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self._reconnect_delay = self.INITIAL_RECONNECT_DELAY

        self.loop.run_until_complete(self._connect())

        self._step_counter = 0
        self.coord_list: list[list[int]] = []

        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise AttributeError("Could not find emulator on wrapped env")

    def step(self, action: int):
        x_pos = self.emulator.memory[PLAYER_X]
        y_pos = self.emulator.memory[PLAYER_Y]
        map_n = self.emulator.memory[MAP_NUMBER]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self._step_counter >= self.upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(self.env.seen_coords)}"
            self.loop.run_until_complete(
                self._send(
                    json.dumps({
                        "metadata": self.stream_metadata,
                        "coords": self.coord_list,
                    })
                )
            )
            self._step_counter = 0
            self.coord_list = []

        self._step_counter += 1
        return self.env.step(action)

    # -- WebSocket helpers with retry ------------------------------------

    async def _connect(self) -> None:
        try:
            import websockets
            self.websocket = await websockets.connect(self.ws_address)
            self._reconnect_delay = self.INITIAL_RECONNECT_DELAY
            logger.debug("WebSocket connected to %s", self.ws_address)
        except Exception as e:
            logger.warning("WebSocket connection failed: %s", e)
            self.websocket = None

    async def _send(self, message: str) -> None:
        import websockets

        if self.websocket is None:
            await self._connect()

        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                logger.warning(
                    "WebSocket send failed, reconnecting in %.1fs",
                    self._reconnect_delay,
                )
                self.websocket = None
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY
                )
                await self._connect()
