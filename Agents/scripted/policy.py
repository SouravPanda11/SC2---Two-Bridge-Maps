"""Pure scripted policy for lower-bridge navigation and Marine micro.

The policy deliberately consumes exact raw unit state, so it is a privileged
state-oracle baseline. Its commands use the benchmark primitives: noop, one of
eight two-world-unit movement directions, or an in-range targeted attack. The
default emits one joint selected-unit action; an optional mode emits per-unit
combat actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal


Point = tuple[float, float]
Side = Literal["left", "right"]
Tactic = Literal[
    "focus_fire",
    "focus_fire_kite",
    "per_unit_focus_fire_kite",
]

NO_OP = 0
MOVE = 1
ATTACK = 2

# These IDs and deltas match MOVE_DIRS in the benchmark Gym wrappers.
MOVE_DELTAS: dict[int, Point] = {
    1: (0.0, -2.0),
    2: (0.0, 2.0),
    3: (-2.0, 0.0),
    4: (2.0, 0.0),
    5: (2.0, -2.0),
    6: (-2.0, -2.0),
    7: (2.0, 2.0),
    8: (-2.0, 2.0),
}

# Live-verified with raw_crop_to_playable_area=True on the 56x62 maps.
# SC2 raw coordinates are Cartesian (low y is the visual bottom), whereas the
# feature-minimap image is vertically inverted. The visual lower bridge is y~17.
LOWER_BRIDGE_LEFT: Point = (16.0, 17.0)
LOWER_BRIDGE_CENTER: Point = (28.0, 17.0)
LOWER_BRIDGE_RIGHT: Point = (36.0, 17.0)


@dataclass(frozen=True)
class UnitSnapshot:
    tag: int
    x: float
    y: float
    health: float
    weapon_cooldown: float = 0.0

    @property
    def position(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class WorldSnapshot:
    friends: tuple[UnitSnapshot, ...]
    enemies: tuple[UnitSnapshot, ...]
    beacon: UnitSnapshot | None
    game_loop: int = 0


@dataclass(frozen=True)
class ScriptedCommand:
    """One benchmark-compatible joint command."""

    verb: int
    unit_tags: tuple[int, ...] = ()
    direction: int = 0
    target_tag: int | None = None
    reason: str = ""

    @classmethod
    def noop(cls, reason: str) -> "ScriptedCommand":
        return cls(verb=NO_OP, reason=reason)


@dataclass(frozen=True)
class AgentConfig:
    tactic: Tactic = "focus_fire_kite"
    waypoint_tolerance: float = 3.25
    waypoint_fraction: float = 0.6
    bridge_log_radius: float = 4.5
    axis_deadband: float = 0.65
    attack_range: float = 6.0
    kite_trigger_range: float = 5.75
    kite_cooldown_fraction: float = 0.5
    wounded_retreat_health: float = 15.0

    def __post_init__(self) -> None:
        if self.tactic not in {
            "focus_fire",
            "focus_fire_kite",
            "per_unit_focus_fire_kite",
        }:
            raise ValueError(f"Unsupported tactic: {self.tactic}")
        if not 0.0 < self.waypoint_fraction <= 1.0:
            raise ValueError("waypoint_fraction must be in (0, 1].")


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def centroid(units: Iterable[UnitSnapshot]) -> Point:
    unit_list = tuple(units)
    if not unit_list:
        raise ValueError("Cannot compute a centroid for an empty unit collection.")
    return (
        sum(unit.x for unit in unit_list) / len(unit_list),
        sum(unit.y for unit in unit_list) / len(unit_list),
    )


def side_of(point: Point) -> Side:
    return "left" if point[0] < LOWER_BRIDGE_CENTER[0] else "right"


def mouth_for(side: Side) -> Point:
    return LOWER_BRIDGE_LEFT if side == "left" else LOWER_BRIDGE_RIGHT


def direction_toward(delta_x: float, delta_y: float, deadband: float = 0.65) -> int:
    """Map a desired vector to the benchmark's nearest compass action."""

    sx = 0 if abs(delta_x) <= deadband else (1 if delta_x > 0 else -1)
    sy = 0 if abs(delta_y) <= deadband else (1 if delta_y > 0 else -1)
    by_sign = {
        (0, -1): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (1, 0): 4,
        (1, -1): 5,
        (-1, -1): 6,
        (1, 1): 7,
        (-1, 1): 8,
    }
    return by_sign.get((sx, sy), 0)


def command_target(snapshot: WorldSnapshot, command: ScriptedCommand) -> Point | None:
    """Translate a discrete movement command to its raw group target point."""

    if command.verb != MOVE or command.direction not in MOVE_DELTAS:
        return None
    selected_tags = set(command.unit_tags)
    selected_friends = tuple(
        friend for friend in snapshot.friends if friend.tag in selected_tags
    )
    if not selected_friends:
        return None
    center = centroid(selected_friends)
    dx, dy = MOVE_DELTAS[command.direction]
    return (center[0] + dx, center[1] + dy)


class LowerBridgeCombatAgent:
    """Finite-state lower-bridge route followed by focus-fire Marine micro."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.route: tuple[Point, ...] = ()
        self.waypoint_index = 0
        self.bridge_reached = False
        self.enemy_contacted = False
        self.first_damage_observed = False
        self.initial_enemy_health = 0.0
        self.focus_target_tag: int | None = None
        self.start_side: Side | None = None
        self.enemy_side: Side | None = None
        self._last_enemy_health = 0.0

    @property
    def route_complete(self) -> bool:
        return bool(self.route) and self.waypoint_index >= len(self.route)

    @property
    def current_waypoint(self) -> Point | None:
        if not self.route or self.route_complete:
            return None
        return self.route[self.waypoint_index]

    def reset(self, snapshot: WorldSnapshot) -> None:
        if not snapshot.friends:
            raise ValueError("The scripted agent needs at least one friendly Marine.")
        if not snapshot.enemies:
            raise ValueError("The scripted agent needs at least one enemy Marine.")

        friend_center = centroid(snapshot.friends)
        enemy_center = centroid(snapshot.enemies)
        self.start_side = side_of(friend_center)
        self.enemy_side = side_of(enemy_center)

        start_mouth = mouth_for(self.start_side)
        enemy_mouth = mouth_for(self.enemy_side)
        # The side corridor is deliberately used before changing y. Spawned
        # beacons sit farther toward the outside edge, so this route avoids an
        # incidental navigation win before the intended combat engagement.
        route = [start_mouth, LOWER_BRIDGE_CENTER]
        if self.enemy_side != self.start_side:
            route.append(enemy_mouth)
        else:
            # Combat maps put both teams on the right. Touch the lower bridge,
            # then leave through the same mouth before approaching the enemy.
            route.append(start_mouth)
        route.append((enemy_mouth[0], enemy_center[1]))

        deduped: list[Point] = []
        for point in route:
            if not deduped or distance(deduped[-1], point) > 0.5:
                deduped.append(point)

        self.route = tuple(deduped)
        self.waypoint_index = 0
        self.bridge_reached = False
        self.enemy_contacted = False
        self.first_damage_observed = False
        self.initial_enemy_health = sum(unit.health for unit in snapshot.enemies)
        self._last_enemy_health = self.initial_enemy_health
        self.focus_target_tag = None

    def _fraction_near(self, units: tuple[UnitSnapshot, ...], point: Point, radius: float) -> float:
        if not units:
            return 0.0
        count = sum(distance(unit.position, point) <= radius for unit in units)
        return count / len(units)

    def observe(self, snapshot: WorldSnapshot) -> None:
        if self._fraction_near(
            snapshot.friends,
            LOWER_BRIDGE_CENTER,
            self.config.bridge_log_radius,
        ) >= self.config.waypoint_fraction:
            self.bridge_reached = True

        total_enemy_health = sum(unit.health for unit in snapshot.enemies)
        if total_enemy_health < self.initial_enemy_health:
            self.first_damage_observed = True
        self._last_enemy_health = total_enemy_health

        if snapshot.friends and snapshot.enemies:
            nearest = min(
                distance(friend.position, enemy.position)
                for friend in snapshot.friends
                for enemy in snapshot.enemies
            )
            if nearest <= self.config.attack_range + 1.0:
                self.enemy_contacted = True

    def _advance_arrived_waypoints(self, snapshot: WorldSnapshot) -> None:
        while not self.route_complete:
            waypoint = self.route[self.waypoint_index]
            if self._fraction_near(
                snapshot.friends,
                waypoint,
                self.config.waypoint_tolerance,
            ) < self.config.waypoint_fraction:
                break
            if distance(waypoint, LOWER_BRIDGE_CENTER) <= 0.5:
                self.bridge_reached = True
            self.waypoint_index += 1

    def _focus_target(self, snapshot: WorldSnapshot) -> UnitSnapshot:
        friends = snapshot.friends
        enemies = snapshot.enemies
        coverage = {
            enemy.tag: sum(
                distance(friend.position, enemy.position) <= self.config.attack_range
                for friend in friends
            )
            for enemy in enemies
        }
        maximum_coverage = max(coverage.values(), default=0)

        if maximum_coverage > 0:
            candidates = tuple(
                enemy for enemy in enemies if coverage[enemy.tag] == maximum_coverage
            )
            target = min(
                candidates,
                key=lambda enemy: (
                    enemy.health,
                    min(distance(friend.position, enemy.position) for friend in friends),
                    enemy.tag,
                ),
            )
        else:
            friend_center = centroid(friends)
            target = min(
                enemies,
                key=lambda enemy: (
                    distance(friend_center, enemy.position),
                    enemy.health,
                    enemy.tag,
                ),
            )
        self.focus_target_tag = target.tag
        return target

    def _kite_direction(self, snapshot: WorldSnapshot) -> int:
        friend_center = centroid(snapshot.friends)
        nearby = tuple(
            enemy
            for enemy in snapshot.enemies
            if distance(friend_center, enemy.position) <= self.config.kite_trigger_range + 2.0
        )
        threat_center = centroid(nearby or snapshot.enemies)
        dx = friend_center[0] - threat_center[0]
        dy = friend_center[1] - threat_center[1]
        if math.hypot(dx, dy) <= 0.1:
            dx = LOWER_BRIDGE_CENTER[0] - friend_center[0]
            dy = LOWER_BRIDGE_CENTER[1] - friend_center[1]
        return direction_toward(dx, dy, self.config.axis_deadband)

    def _route_command(self, snapshot: WorldSnapshot) -> ScriptedCommand:
        waypoint = self.route[self.waypoint_index]
        center = centroid(snapshot.friends)
        direction = direction_toward(
            waypoint[0] - center[0],
            waypoint[1] - center[1],
            self.config.axis_deadband,
        )
        if direction == 0:
            # A dispersed squad can have its centroid at the target before
            # enough individual units satisfy the arrival fraction.
            farthest = max(
                snapshot.friends,
                key=lambda unit: distance(unit.position, waypoint),
            )
            direction = direction_toward(
                waypoint[0] - farthest.x,
                waypoint[1] - farthest.y,
                self.config.axis_deadband,
            )
        if direction == 0:
            return ScriptedCommand.noop("waiting_for_waypoint_formation")
        return ScriptedCommand(
            verb=MOVE,
            unit_tags=tuple(sorted(unit.tag for unit in snapshot.friends)),
            direction=direction,
            reason=f"route_to_waypoint_{self.waypoint_index}",
        )

    def _joint_combat_command(self, snapshot: WorldSnapshot) -> ScriptedCommand:
        target = self._focus_target(snapshot)
        attackable = tuple(
            friend
            for friend in snapshot.friends
            if distance(friend.position, target.position) <= self.config.attack_range
        )
        if not attackable:
            center = centroid(snapshot.friends)
            direction = direction_toward(
                target.x - center[0],
                target.y - center[1],
                self.config.axis_deadband,
            )
            return ScriptedCommand(
                verb=MOVE if direction else NO_OP,
                unit_tags=tuple(sorted(unit.tag for unit in snapshot.friends)),
                direction=direction,
                reason="joint_approach_target" if direction else "joint_hold",
            )

        tags = tuple(sorted(unit.tag for unit in attackable))
        nearest_enemy_distance = min(
            distance(friend.position, enemy.position)
            for friend in attackable
            for enemy in snapshot.enemies
        )
        cooling_fraction = sum(
            friend.weapon_cooldown > 0.0 for friend in attackable
        ) / len(attackable)

        if (
            self.config.tactic == "focus_fire_kite"
            and nearest_enemy_distance <= self.config.kite_trigger_range
            and cooling_fraction >= self.config.kite_cooldown_fraction
        ):
            direction = self._kite_direction(snapshot)
            if direction:
                return ScriptedCommand(
                    verb=MOVE,
                    unit_tags=tags,
                    direction=direction,
                    reason="cooldown_kite",
                )

        return ScriptedCommand(
            verb=ATTACK,
            unit_tags=tags,
            target_tag=target.tag,
            reason="focus_fire",
        )

    def _per_unit_combat_commands(
        self,
        snapshot: WorldSnapshot,
    ) -> tuple[ScriptedCommand, ...]:
        """Return one legal 9+E primitive action for every living Marine."""

        target = self._focus_target(snapshot)
        commands: list[ScriptedCommand] = []
        for friend in snapshot.friends:
            target_distance = distance(friend.position, target.position)
            nearest_enemy = min(
                snapshot.enemies,
                key=lambda enemy: distance(friend.position, enemy.position),
            )
            nearest_distance = distance(friend.position, nearest_enemy.position)
            should_retreat = (
                nearest_distance <= self.config.kite_trigger_range
                and (
                    friend.weapon_cooldown > 0.0
                    or friend.health <= self.config.wounded_retreat_health
                )
            )

            if should_retreat:
                dx = friend.x - nearest_enemy.x
                dy = friend.y - nearest_enemy.y
                if math.hypot(dx, dy) <= 0.1:
                    dx = LOWER_BRIDGE_CENTER[0] - friend.x
                    dy = LOWER_BRIDGE_CENTER[1] - friend.y
                direction = direction_toward(dx, dy, self.config.axis_deadband)
                if direction:
                    commands.append(
                        ScriptedCommand(
                            verb=MOVE,
                            unit_tags=(friend.tag,),
                            direction=direction,
                            reason="per_unit_cooldown_kite",
                        )
                    )
                    continue

            if target_distance <= self.config.attack_range:
                commands.append(
                    ScriptedCommand(
                        verb=ATTACK,
                        unit_tags=(friend.tag,),
                        target_tag=target.tag,
                        reason="per_unit_focus_fire",
                    )
                )
                continue

            direction = direction_toward(
                target.x - friend.x,
                target.y - friend.y,
                self.config.axis_deadband,
            )
            commands.append(
                ScriptedCommand(
                    verb=MOVE if direction else NO_OP,
                    unit_tags=(friend.tag,),
                    direction=direction,
                    reason="per_unit_approach_target" if direction else "per_unit_hold",
                )
            )
        return tuple(commands)

    def act_many(self, snapshot: WorldSnapshot) -> tuple[ScriptedCommand, ...]:
        if not snapshot.friends:
            return (ScriptedCommand.noop("no_friendly_marines"),)
        if not snapshot.enemies:
            return (ScriptedCommand.noop("no_enemy_marines"),)
        if not self.route:
            self.reset(snapshot)

        self.observe(snapshot)
        self._advance_arrived_waypoints(snapshot)

        if not self.route_complete:
            return (self._route_command(snapshot),)
        if self.config.tactic == "per_unit_focus_fire_kite":
            return self._per_unit_combat_commands(snapshot)
        return (self._joint_combat_command(snapshot),)

    def act(self, snapshot: WorldSnapshot) -> ScriptedCommand:
        """Return the first command; use act_many for per-unit combat micro."""

        return self.act_many(snapshot)[0]
