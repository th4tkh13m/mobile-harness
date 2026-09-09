from __future__ import annotations

import unittest
from unittest.mock import mock_open, patch
import os
import json
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from mobile_harness.core import Harness
from mobile_harness.model import Action, ActionKind, ActionResult, Decision, Observation, Rect, UIElement
from mobile_harness.policy import DevicePolicy


class FakeDevice:
    def __init__(self, sensitive: bool = False) -> None:
        text = "Allow this permission" if sensitive else "Continue"
        self.observation = Observation(100, 200, elements=(UIElement(Rect(1, 1, 30, 20), text=text, clickable=True),))
        self.actions: list[Action] = []

    def observe(self) -> Observation:
        return self.observation

    def act(self, action: Action, observation: Observation) -> ActionResult:
        self.actions.append(action)
        return ActionResult(True)


class SequenceAgent:
    def __init__(self, decisions: list[Decision]) -> None:
        self.decisions = iter(decisions)

    def decide(self, task, observation, history):
        return next(self.decisions)


class Verifier:
    def __init__(self, answers: list[bool]) -> None:
        self.answers = iter(answers)

    def verify(self, task, device, observation, history):
        ok = next(self.answers)
        return ok, "verified" if ok else "not yet"


class HarnessTests(unittest.TestCase):
    def test_failed_terminal_claim_keeps_loop_running_until_verification_passes(self):
        device = FakeDevice()
        agent = SequenceAgent([
            Decision(done=True, reason="first claim"),
            Decision(action=Action(ActionKind.TAP, x=.2, y=.2)),
            Decision(done=True, reason="second claim"),
        ])
        result = Harness(device, Verifier([False, True]), max_steps=3).run("task", agent)
        self.assertTrue(result.success)
        self.assertEqual([action.kind for action in device.actions], [ActionKind.TAP])

    def test_sensitive_screen_blocks_interaction_without_explicit_approval(self):
        from mobile_harness.config import PolicySettings
        device = FakeDevice(sensitive=True)
        agent = SequenceAgent([Decision(action=Action(ActionKind.TAP, x=.2, y=.2))])
        result = Harness(device, Verifier([]), policy=DevicePolicy(settings=PolicySettings(("permission",), ("tap",))), max_steps=1).run("task", agent)
        self.assertFalse(result.success)
        self.assertEqual(device.actions, [])
        self.assertIn("sensitive UI", result.events[0].policy_message)

    def test_policy_settings_are_loaded_from_central_json_config(self):
        from mobile_harness.config import load_harness_config
        with TemporaryDirectory() as directory:
            config = Path(directory) / "policy.json"
            config.write_text(json.dumps({
                "model": {"name": None, "base_url": None},
                "device": {"serial": None, "adb_timeout_seconds": 10},
                "harness": {"max_steps": 5, "trace_path": "trace.jsonl", "zoom_ratio": .5},
                "runtime": {"provider": "chat_completions", "max_turns": 5, "session_root": "sessions", "workspace_root": ".", "capture_after_actions": True, "require_plan_before_actions": True, "require_plan_update_on_transition": True, "context_window_tokens": 200000, "memory_root": "memory", "skills_root": "skills", "compaction_summary_timeout_seconds": 20, "model_stale_timeout_seconds": 150, "model_streaming": True, "model_stream_progress_interval_seconds": 15},
                "authority": {"auto_approve_mobile_actions": True},
                "policy": {"sensitive_terms": ["password"], "guarded_action_kinds": ["type_text"]},
            }))
            settings = load_harness_config(config)
            self.assertEqual(settings.harness.max_steps, 5)
            self.assertEqual(settings.device.adb_timeout_seconds, 10)
            self.assertTrue(settings.runtime.capture_after_actions)
            self.assertTrue(settings.runtime.require_plan_before_actions)
            self.assertTrue(settings.runtime.require_plan_update_on_transition)
            self.assertEqual(settings.runtime.context_window_tokens, 200000)
            self.assertTrue(settings.authority.auto_approve_mobile_actions)
            decision = DevicePolicy(settings=settings.policy).check(
                Action(ActionKind.TAP, x=.2, y=.2), FakeDevice(sensitive=True).observe()
            )
        self.assertTrue(decision.allowed)

    def test_locator_is_preferred_over_coordinates(self):
        from mobile_harness.ports import resolve_tap
        observation = Observation(100, 200, elements=(UIElement(Rect(10, 20, 30, 60), text="Open"),))
        self.assertEqual(resolve_tap(Action(ActionKind.TAP, locator={"text": "Open"}), observation), (20, 40))

    def test_harness_resolves_locator_before_any_device_port(self):
        device = FakeDevice()
        Harness(device, verifier=Verifier([True]), max_steps=1).run(
            "tap", SequenceAgent([Decision(action=Action(ActionKind.TAP, locator={"text": "Continue"}))])
        )
        action = device.actions[0]
        self.assertEqual(action.locator, {"text": "Continue"})
        self.assertEqual((action.x, action.y), (.15, .05))

    def test_adb_port_never_constructs_an_arbitrary_shell_command(self):
        from mobile_harness.ports import AdbDevice
        observation = Observation(100, 200)
        argv = AdbDevice()._action_argv(Action(ActionKind.TAP, x=.5, y=.25), observation)
        self.assertEqual(argv, ("shell", "input", "tap", "50", "50"))
        with self.assertRaises(ValueError):
            AdbDevice()._action_argv(Action(ActionKind.TYPE_TEXT, text="xin chào"), observation)

    def test_key_is_the_only_back_and_home_transport(self):
        from mobile_harness.contracts import openai_tools
        from mobile_harness.ports import AdbDevice
        names = {tool["function"]["name"] for tool in openai_tools()}
        self.assertNotIn("back", names)
        self.assertNotIn("home", names)
        observation = Observation(100, 200)
        self.assertEqual(AdbDevice()._action_argv(Action(ActionKind.KEY, key="BACK"), observation), ("shell", "input", "keyevent", "BACK"))
        self.assertEqual(AdbDevice()._action_argv(Action(ActionKind.KEY, key="HOME"), observation), ("shell", "input", "keyevent", "HOME"))

    def test_key_schema_and_transports_share_the_explicit_key_vocabulary(self):
        from mobile_harness.contracts import decision_from_tool_call, openai_tools
        from mobile_harness.ports import AdbDevice
        key_schema = next(tool["function"] for tool in openai_tools() if tool["function"]["name"] == "key")
        self.assertIn("BACK", key_schema["parameters"]["properties"]["key"]["enum"])
        self.assertIn("VOLUME_MUTE", key_schema["parameters"]["properties"]["key"]["enum"])
        with self.assertRaisesRegex(ValueError, "unsupported key"):
            decision_from_tool_call("key", {"key": "KEYCODE_ENTER"})
        with self.assertRaisesRegex(ValueError, "unsupported key"):
            AdbDevice()._action_argv(Action(ActionKind.KEY, key="POWER"), Observation(100, 200))

    def test_model_action_contract_uses_integer_1000_space_but_internal_actions_remain_normalized(self):
        from mobile_harness.contracts import decision_from_tool_call, openai_tools
        tap_schema = next(tool["function"] for tool in openai_tools() if tool["function"]["name"] == "tap")
        self.assertEqual(tap_schema["parameters"]["properties"]["x"]["type"], "integer")
        self.assertEqual(tap_schema["parameters"]["properties"]["x"]["maximum"], 1000)
        decision = decision_from_tool_call("tap", {"x": 500, "y": 250})
        self.assertEqual((decision.action.x, decision.action.y), (.5, .25))
        with self.assertRaisesRegex(ValueError, "integer"):
            decision_from_tool_call("tap", {"x": .5, "y": .25})

    def test_zoom_crops_observation_and_maps_following_tap_to_physical_screen(self):
        from PIL import Image
        from mobile_harness.ports import AdbDevice
        image = Image.new("RGB", (100, 200), "white")
        encoded = io.BytesIO(); image.save(encoded, format="PNG")
        observation = Observation(100, 200, screenshot_png=encoded.getvalue(), elements=(UIElement(Rect(45, 95, 55, 105), text="Target", clickable=True),))
        zoom = observation.zoomed(.5, .5, .5)
        self.assertEqual((zoom.width, zoom.height, zoom.viewport_left, zoom.viewport_top), (50, 100, 25, 50))
        self.assertEqual(zoom.elements[0].bounds, Rect(20, 45, 30, 55))
        self.assertEqual(
            AdbDevice()._action_argv(Action(ActionKind.TAP, x=.5, y=.5), zoom),
            ("shell", "input", "tap", "50", "100"),
        )

    def test_zoom_is_a_view_only_action_before_the_following_device_action(self):
        from PIL import Image
        image = Image.new("RGB", (100, 200), "white")
        encoded = io.BytesIO(); image.save(encoded, format="PNG")
        device = FakeDevice()
        device.observation = Observation(100, 200, screenshot_png=encoded.getvalue())
        agent = SequenceAgent([
            Decision(action=Action(ActionKind.ZOOM, x=.5, y=.5)),
            Decision(action=Action(ActionKind.TAP, x=.5, y=.5)),
            Decision(done=True),
        ])
        result = Harness(device, Verifier([True]), max_steps=3).run("zoom then tap", agent)
        self.assertTrue(result.success)
        self.assertEqual([event.result.message for event in result.events[:2]], ["zoomed to 50x100", ""])
        self.assertEqual([action.kind for action in device.actions], [ActionKind.TAP])

    def test_adb_path_discovery_supports_android_sdk_root(self):
        from mobile_harness.ports import resolve_adb_path
        with patch("shutil.which", return_value=None), patch.dict(os.environ, {"ANDROID_HOME": "C:/sdk", "ANDROID_SDK_ROOT": ""}, clear=False), patch("pathlib.Path.is_file", return_value=True):
            self.assertEqual(resolve_adb_path(), "C:\\sdk\\platform-tools\\adb.exe")

    def test_tool_contract_rejects_unknown_tools_and_trace_is_jsonl(self):
        from mobile_harness.contracts import decision_from_tool_call
        from mobile_harness.trace import JsonlTrace
        with self.assertRaises(ValueError):
            decision_from_tool_call("shell", {"command": "id"})
        sink = mock_open()
        with patch("pathlib.Path.open", sink):
            result = Harness(FakeDevice(), Verifier([True]), trace=JsonlTrace("trace.jsonl"), max_steps=1).run("task", SequenceAgent([Decision(done=True)]))
        self.assertTrue(result.success)
        self.assertIn('"step": 1', sink().write.call_args.args[0])

    def test_mai_ui_translation_and_verified_memory_promotion(self):
        from mobile_harness.mai_ui import decision_from_mai_action
        from mobile_harness.memory import Experience, VerifiedExperienceStore
        decision = decision_from_mai_action({"action": "click", "coordinate": [.3, .4]})
        self.assertEqual((decision.action.kind, decision.action.x, decision.action.y), (ActionKind.TAP, .3, .4))
        store = VerifiedExperienceStore("unused.jsonl")
        with self.assertRaises(PermissionError):
            store.promote(Experience("task", "summary", "evidence"), verifier_passed=False)

    def test_openai_adapter_requires_one_strict_mobile_tool_call(self):
        from mobile_harness.openai_agent import OpenAICompatibleAgent
        response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "tap", "arguments": '{"locator":{"text":"Continue"}}'}}]}}]}
        agent = OpenAICompatibleAgent("https://example.invalid/v1", "key", "model", post=lambda *_: response)
        decision = agent.decide("continue", FakeDevice().observe(), ())
        self.assertEqual(decision.action.locator, {"text": "Continue"})

    def test_mobileworld_verifier_uses_its_authoritative_score(self):
        from mobile_harness.benchmarks import MobileWorldVerifier
        class Environment:
            def get_task_score(self, task_type):
                self.task_type = task_type
                return .5, "backend state matched"
        environment = Environment()
        passed, message = MobileWorldVerifier(environment, "task_1").verify("irrelevant", FakeDevice(), FakeDevice().observe(), ())
        self.assertTrue(passed)
        self.assertEqual(environment.task_type, "task_1")
        self.assertIn("backend state matched", message)

    def test_memgui_verifier_uses_the_compatible_runtime_score(self):
        from mobile_harness.benchmarks import MemGUIVerifier
        class Environment:
            def get_task_score(self, task_type):
                return 1.0, f"{task_type} passed progressive scrutiny"
        passed, message = MemGUIVerifier(Environment(), "memory_task").verify("irrelevant", FakeDevice(), FakeDevice().observe(), ())
        self.assertTrue(passed)
        self.assertIn("MemGUI score=1.0", message)

    def test_androidworld_adapter_passes_current_dimensions_to_action_factory_and_uses_task_score(self):
        from mobile_harness.benchmarks import AndroidWorldAdapter, AndroidWorldVerifier
        class Box:
            x_min = 10
            y_min = 20
            x_max = 30
            y_max = 60
        class Element:
            bbox_pixels = Box()
            text = "Continue"
            content_description = ""
        class State:
            pixels = None
            ui_elements = [Element()]
        class Environment:
            logical_screen_size = (100, 200)
            foreground_activity_name = "pkg/.Main"
            controller = object()
            def get_state(self, wait_to_stabilize):
                return State()
        captured = {}
        def factory(action, width, height):
            captured.update(width=width, height=height)
            return "converted"
        def executor(action, elements, size, controller):
            captured.update(action=action, size=size)
        env = Environment()
        adapter = AndroidWorldAdapter(env, executor, factory)
        observation = adapter.observe()
        adapter.act(Action(ActionKind.TAP, x=.2, y=.2), observation)
        self.assertEqual(captured, {"width": 100, "height": 200, "action": "converted", "size": (100, 200)})
        class Task:
            def is_successful(self, active_env):
                self.active_env = active_env
                return 1.0
        task = Task()
        passed, _ = AndroidWorldVerifier(task, env).verify("task", adapter, observation, ())
        self.assertTrue(passed)
        self.assertIs(task.active_env, env)

    def test_foreground_activity_keeps_only_the_focused_component(self):
        from mobile_harness.ports import foreground_activity
        dump = "mCurrentFocus=Window{123 u0 com.example/.MainActivity}\nadditional dumpsys text"
        self.assertEqual(foreground_activity(dump), "com.example/.MainActivity")

    def test_mobileworld_runner_owns_task_lifecycle(self):
        from mobile_harness import runners
        class Environment:
            def get_task_goal(self, task_type):
                self.goal_task = task_type
                return "complete it"
            def initialize_task(self, task_name):
                self.initialized = task_name
            def tear_down_task(self, task_type):
                self.torn_down = task_type
        env = Environment()
        sentinel = object()
        with patch.object(runners.Harness, "run", return_value=sentinel) as run:
            result = runners.run_mobileworld_task(env, "demo", object())
        self.assertIs(result, sentinel)
        self.assertEqual((env.goal_task, env.initialized, env.torn_down), ("demo", "demo", "demo"))
        self.assertEqual(run.call_args.args[0], "complete it")


if __name__ == "__main__":
    unittest.main()
