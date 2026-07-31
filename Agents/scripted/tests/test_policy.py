from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Agents.scripted._evaluate import (
    classify_outcome,
    infer_region,
    raw_actions_for_command,
    resolve_variant_map_file,
    wilson_interval,
)
from Agents.scripted.policy import (
    ATTACK,
    MOVE,
    AgentConfig,
    LOWER_BRIDGE_CENTER,
    LOWER_BRIDGE_LEFT,
    LOWER_BRIDGE_RIGHT,
    LowerBridgeCombatAgent,
    ScriptedCommand,
    UnitSnapshot,
    WorldSnapshot,
    command_target,
)
from Agents.scripted.variants import MAP_VARIANTS


def squad(center_x: float, center_y: float, *, cooldown: float = 0.0):
    offsets = ((-1.0, 0.0), (0.0, -1.0), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
    return tuple(
        UnitSnapshot(
            tag=index + 1,
            x=center_x + dx,
            y=center_y + dy,
            health=45.0,
            weapon_cooldown=cooldown,
        )
        for index, (dx, dy) in enumerate(offsets)
    )


def enemies(center_x: float, center_y: float, count: int = 3):
    return tuple(
        UnitSnapshot(
            tag=100 + index,
            x=center_x,
            y=center_y + index,
            health=45.0,
        )
        for index in range(count)
    )


def state(friend_xy, enemy_xy, *, cooldown: float = 0.0, enemy_count: int = 3):
    return WorldSnapshot(
        friends=squad(*friend_xy, cooldown=cooldown),
        enemies=enemies(*enemy_xy, count=enemy_count),
        beacon=UnitSnapshot(tag=900, x=50.0, y=5.0, health=1.0),
    )


class PolicyRouteTests(unittest.TestCase):
    def test_base_route_uses_visual_lower_bridge_left_to_right(self):
        initial = state((5.0, 58.0), (50.0, 35.0))
        agent = LowerBridgeCombatAgent(AgentConfig(tactic="focus_fire"))
        agent.reset(initial)

        self.assertEqual(
            agent.route,
            (LOWER_BRIDGE_LEFT, LOWER_BRIDGE_CENTER, LOWER_BRIDGE_RIGHT, (36.0, 36.0)),
        )
        first_commands = agent.act_many(initial)
        self.assertTrue(all(command.verb == MOVE for command in first_commands))
        self.assertEqual(
            {tag for command in first_commands for tag in command.unit_tags},
            {1, 2, 3, 4, 5},
        )

        at_left = state(LOWER_BRIDGE_LEFT, (50.0, 35.0))
        self.assertEqual(agent.act(at_left).direction, 4)  # east through the lower gap

        at_center = state(LOWER_BRIDGE_CENTER, (50.0, 35.0))
        self.assertEqual(agent.act(at_center).direction, 4)
        self.assertTrue(agent.bridge_reached)

        at_right = state(LOWER_BRIDGE_RIGHT, (50.0, 35.0))
        self.assertEqual(agent.act(at_right).direction, 2)  # north in the beacon-safe corridor

        staged = state((36.0, 36.0), (50.0, 35.0))
        self.assertEqual(agent.act(staged).verb, MOVE)
        self.assertTrue(agent.route_complete)

        in_range = state((45.0, 36.0), (50.0, 35.0))
        self.assertEqual(agent.act(in_range).verb, ATTACK)

    def test_combat_route_touches_bridge_then_returns_to_enemy_side(self):
        initial = state((50.0, 58.0), (50.0, 35.0))
        agent = LowerBridgeCombatAgent()
        agent.reset(initial)
        self.assertEqual(
            agent.route,
            (LOWER_BRIDGE_RIGHT, LOWER_BRIDGE_CENTER, LOWER_BRIDGE_RIGHT, (36.0, 36.0)),
        )

    def test_navigate_route_reverses_right_to_left(self):
        initial = state((50.0, 58.0), (5.0, 5.0))
        agent = LowerBridgeCombatAgent()
        agent.reset(initial)
        self.assertEqual(
            agent.route,
            (LOWER_BRIDGE_RIGHT, LOWER_BRIDGE_CENTER, LOWER_BRIDGE_LEFT, (16.0, 6.0)),
        )

class PolicyCombatTests(unittest.TestCase):
    def test_focus_fire_prefers_low_health_target_in_range(self):
        friends = squad(40.0, 20.0)
        enemy_units = (
            UnitSnapshot(tag=101, x=44.0, y=20.0, health=45.0),
            UnitSnapshot(tag=102, x=44.0, y=21.0, health=7.0),
            UnitSnapshot(tag=103, x=50.0, y=20.0, health=1.0),
        )
        snapshot = WorldSnapshot(friends, enemy_units, None)
        agent = LowerBridgeCombatAgent(AgentConfig(tactic="focus_fire"))
        agent.reset(snapshot)
        agent.waypoint_index = len(agent.route)

        command = agent.act(snapshot)
        self.assertEqual(command.verb, ATTACK)
        self.assertEqual(command.target_tag, 102)
        self.assertEqual(command.unit_tags, (1, 2, 3, 4, 5))

    def test_kite_uses_one_joint_discrete_move_while_weapons_cool(self):
        snapshot = state((40.0, 20.0), (44.0, 20.0), cooldown=8.0)
        agent = LowerBridgeCombatAgent(AgentConfig(tactic="focus_fire_kite"))
        agent.reset(snapshot)
        agent.waypoint_index = len(agent.route)

        command = agent.act(snapshot)
        self.assertEqual(command.verb, MOVE)
        self.assertEqual(command.reason, "cooldown_kite")
        self.assertEqual(command.unit_tags, (1, 2, 3, 4, 5))

    def test_move_translation_is_exactly_one_benchmark_step(self):
        snapshot = state((10.0, 10.0), (50.0, 35.0))
        command = ScriptedCommand(verb=MOVE, unit_tags=(1, 2, 3, 4, 5), direction=7)
        self.assertEqual(command_target(snapshot, command), (12.0, 12.0))

    def test_per_unit_micro_emits_one_primitive_action_per_marine(self):
        friends = tuple(
            UnitSnapshot(
                tag=index + 1,
                x=40.0,
                y=19.0 + index * 0.5,
                health=45.0,
                weapon_cooldown=8.0 if index < 3 else 0.0,
            )
            for index in range(5)
        )
        snapshot = WorldSnapshot(friends, enemies(44.0, 20.0), None)
        agent = LowerBridgeCombatAgent(
            AgentConfig(tactic="per_unit_focus_fire_kite")
        )
        agent.reset(snapshot)
        agent.waypoint_index = len(agent.route)

        commands = agent.act_many(snapshot)
        self.assertEqual(len(commands), 5)
        self.assertEqual({tag for command in commands for tag in command.unit_tags}, {1, 2, 3, 4, 5})
        self.assertTrue(any(command.verb == MOVE for command in commands))
        self.assertTrue(any(command.verb == ATTACK for command in commands))


class EvaluatorHelpersTests(unittest.TestCase):
    class _TimeStep:
        reward = 0.0

        @staticmethod
        def last():
            return False

    def test_terminal_classification_separates_combat_and_navigation(self):
        no_enemies = WorldSnapshot(squad(5.0, 5.0), (), None)
        self.assertEqual(
            classify_outcome(self._TimeStep(), no_enemies, 1, 600),
            "combat_win",
        )

        beacon = UnitSnapshot(tag=900, x=8.0, y=5.0, health=1.0)
        nav = WorldSnapshot(squad(5.0, 5.0), enemies(50.0, 35.0), beacon)
        self.assertEqual(classify_outcome(self._TimeStep(), nav, 1, 600), "nav_win")

    def test_native_terminal_has_benchmark_priority_over_beacon(self):
        class TerminalTimeStep:
            reward = 1.0

            @staticmethod
            def last():
                return True

        beacon = UnitSnapshot(tag=900, x=5.0, y=5.0, health=1.0)
        snapshot = WorldSnapshot(squad(5.0, 5.0), (), beacon)
        self.assertEqual(
            classify_outcome(TerminalTimeStep(), snapshot, 1, 600),
            "combat_win",
        )

    def test_map_resolution_accepts_repository_filename_case(self):
        variant = MAP_VARIANTS["V3_Navigate"]
        with TemporaryDirectory() as temporary_dir:
            actual = Path(temporary_dir) / "TwoBridgeMap_V3_navigate.SC2Map"
            actual.touch()
            self.assertEqual(
                resolve_variant_map_file(Path(temporary_dir), variant),
                actual,
            )

    def test_region_labels_match_map_figure(self):
        self.assertEqual(infer_region((5.0, 58.0)), "R1")
        self.assertEqual(infer_region((5.0, 35.0)), "R2")
        self.assertEqual(infer_region((5.0, 5.0)), "R3")
        self.assertEqual(infer_region((50.0, 58.0)), "R4")
        self.assertEqual(infer_region((50.0, 35.0)), "R5")
        self.assertEqual(infer_region((50.0, 5.0)), "R6")

    def test_all_nine_variants_and_enemy_counts_are_present(self):
        self.assertEqual(len(MAP_VARIANTS), 9)
        self.assertEqual(MAP_VARIANTS["V1_Base"].enemy_count, 3)
        self.assertEqual(MAP_VARIANTS["V2_Combat"].enemy_count, 5)
        self.assertEqual(MAP_VARIANTS["V3_Navigate"].enemy_count, 8)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(7, 10)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)

    def test_group_route_keeps_all_five_tags_in_one_move(self):
        class RawFunctions:
            @staticmethod
            def Move_pt(queue, tags, point):
                return ("move", queue, tuple(tags), tuple(point))

            @staticmethod
            def Attack_unit(queue, tags, target):
                return ("attack", queue, tuple(tags), target)

        snapshot = state((10.0, 10.0), (50.0, 35.0))
        command = ScriptedCommand(
            verb=MOVE,
            unit_tags=(1, 2, 3, 4, 5),
            direction=4,
        )
        actions = raw_actions_for_command(
            RawFunctions,
            command,
            snapshot,
            (56.0, 62.0),
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0][2], (1, 2, 3, 4, 5))
        self.assertEqual(actions[0][3], (12.0, 10.0))

    def test_joint_attack_keeps_selected_marines_in_one_action(self):
        class RawFunctions:
            @staticmethod
            def Move_pt(queue, tags, point):
                return ("move", queue, tuple(tags), tuple(point))

            @staticmethod
            def Attack_unit(queue, tags, target):
                return ("attack", queue, tuple(tags), target)

        snapshot = state((40.0, 20.0), (44.0, 20.0))
        command = ScriptedCommand(
            verb=ATTACK,
            unit_tags=(1, 2, 3, 4, 5),
            target_tag=100,
        )
        actions = raw_actions_for_command(
            RawFunctions,
            command,
            snapshot,
            (56.0, 62.0),
        )
        self.assertEqual(
            actions,
            [("attack", "now", (1, 2, 3, 4, 5), 100)],
        )


if __name__ == "__main__":
    unittest.main()
