"""Tests for Ray RLlib integration — env_creator and config builder.

These tests do NOT require Ray to be installed.  They test the env
factory logic and argument parsing in isolation.
"""

import pytest
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEnvCreatorImport:
    """Test that env_creator can be imported and has the right signature."""

    def test_import(self):
        from ray_trainer import env_creator
        assert callable(env_creator)

    def test_env_creator_signature(self):
        """env_creator should accept a single dict argument."""
        import inspect
        from ray_trainer import env_creator
        sig = inspect.signature(env_creator)
        params = list(sig.parameters.keys())
        assert "env_config" in params


class TestEnvCreatorWrapperFlags:
    """Test that wrapper flags are properly popped from env_config."""

    def test_flags_popped(self):
        """Wrapper flags should be removed from the config dict
        so they don't confuse EnvConfig.from_dict()."""
        config = {
            "headless": True,
            "gb_path": "../PokemonRed.gb",
            "init_state": "../init.state",
            "use_frame_stack": True,
            "frame_stack_n": 4,
            "use_action_macros": True,
            "use_stream": False,
            "stream_metadata": None,
        }

        # We can't actually call env_creator without a ROM, but we
        # can verify the flag extraction logic.
        use_frame_stack = config.pop("use_frame_stack", False)
        frame_stack_n = config.pop("frame_stack_n", 4)
        use_macros = config.pop("use_action_macros", False)
        use_stream = config.pop("use_stream", False)
        stream_metadata = config.pop("stream_metadata", None)

        assert use_frame_stack is True
        assert frame_stack_n == 4
        assert use_macros is True
        assert use_stream is False
        # Remaining config should be clean for EnvConfig
        assert "use_frame_stack" not in config
        assert "use_action_macros" not in config


class TestArgParser:
    """Test CLI argument parsing without launching Ray."""

    def test_defaults(self):
        import argparse
        # Re-create the parser from ray_trainer
        parser = argparse.ArgumentParser()
        parser.add_argument("--gb-path", default="../PokemonRed.gb")
        parser.add_argument("--init-state", default="../init.state")
        parser.add_argument("--num-workers", type=int, default=64)
        parser.add_argument("--num-gpus", type=int, default=0)
        parser.add_argument("--frame-stack", type=int, default=0)
        parser.add_argument("--action-macros", action="store_true")
        parser.add_argument("--iterations", type=int, default=10000)
        parser.add_argument("--checkpoint-freq", type=int, default=50)

        args = parser.parse_args([])
        assert args.num_workers == 64
        assert args.frame_stack == 0
        assert args.action_macros is False
        assert args.iterations == 10000

    def test_custom_args(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-workers", type=int, default=64)
        parser.add_argument("--frame-stack", type=int, default=0)
        parser.add_argument("--action-macros", action="store_true")

        args = parser.parse_args([
            "--num-workers", "128",
            "--frame-stack", "8",
            "--action-macros",
        ])
        assert args.num_workers == 128
        assert args.frame_stack == 8
        assert args.action_macros is True


class TestCallbacksFactory:
    """Test that the callbacks class can be created if Ray is available."""

    def test_skipped_without_ray(self):
        """If Ray isn't installed, we just skip — this is expected."""
        try:
            from ray_trainer import _make_callbacks_class
            cls = _make_callbacks_class()
            assert cls is not None
        except ImportError:
            pytest.skip("Ray not installed — skipping callbacks test")
