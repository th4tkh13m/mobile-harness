from __future__ import annotations

import unittest
from unittest.mock import mock_open, patch
import os

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
        device = FakeDevice(sensitive=True)
        agent = SequenceAgent([Decision(action=Action(ActionKind.TAP, x=.2, y=.2))])
        result = Harness(device, Verifier([]), policy=DevicePolicy(), max_steps=1).run("task", agent)
        self.assertFalse(result.success)
        self.assertEqual(device.actions, [])
        self.assertIn("sensitive UI", result.events[0].policy_message)

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
