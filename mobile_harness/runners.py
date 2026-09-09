"""Benchmark lifecycle glue kept outside the portable harness core.

The core never creates a task, resets a snapshot, or tears a task down.  Those
operations are benchmark authority.  These helpers make that boundary explicit
and ensure every terminal claim is still checked by the benchmark evaluator.
"""
from __future__ import annotations

from typing import Any

from .benchmarks import (
    AndroidWorldAdapter,
    AndroidWorldVerifier,
    MemGUIAdapter,
    MemGUIVerifier,
    MobileWorldAdapter,
    MobileWorldVerifier,
    androidworld_action_factory,
)
from .core import Agent, Harness, HarnessResult


def run_androidworld_task(env: Any, task_eval: Any, agent: Agent, *, max_steps: int | None = None) -> HarnessResult:
    """Initialize one AndroidWorld task and run it with canonical evaluation."""
    from android_world.env import actuation

    task_eval.initialize_task(env)
    budget = max_steps if max_steps is not None else max(1, int(task_eval.complexity * 10))
    device = AndroidWorldAdapter(env, actuation.execute_adb_action, androidworld_action_factory)
    return Harness(device, AndroidWorldVerifier(task_eval, env), max_steps=budget).run(str(task_eval.goal), agent)


def run_mobileworld_task(env: Any, task_name: str, agent: Agent, *, max_steps: int = 15) -> HarnessResult:
    """Initialize, run, score, and tear down one MobileWorld task."""
    goal = env.get_task_goal(task_type=task_name)
    env.initialize_task(task_name=task_name)
    try:
        return Harness(
            MobileWorldAdapter(env), MobileWorldVerifier(env, task_name), max_steps=max_steps,
        ).run(goal, agent)
    finally:
        env.tear_down_task(task_type=task_name)


def run_memgui_task(env: Any, task_name: str, agent: Agent, *, max_steps: int = 15) -> HarnessResult:
    """Run a MemGUI task through its MobileWorld-compatible environment API."""
    goal = env.get_task_goal(task_type=task_name)
    env.initialize_task(task_name=task_name)
    try:
        return Harness(
            MemGUIAdapter(env), MemGUIVerifier(env, task_name), max_steps=max_steps,
        ).run(goal, agent)
    finally:
        env.tear_down_task(task_type=task_name)
