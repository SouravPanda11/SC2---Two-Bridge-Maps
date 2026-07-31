# Scripted lower-bridge combat baseline

This folder contains one shared finite-state controller and a launcher for each
of the nine Two-Bridge variants. The controller:

1. commands every living friendly Marine together;
2. stages at the near mouth of the **visual lower bridge**;
3. crosses/touches its center and exits on the required side;
4. approaches the enemy along a side corridor that reduces accidental beacon
   captures; and
5. focus-fires one enemy at a time and briefly kites the selected group while
   most Marine weapons are on cooldown.

In PySC2's cropped RAW Cartesian coordinates, the verified lower route is
`(16, 17) -> (28, 17) -> (36, 17)`. The feature-minimap image is vertically
inverted, which is why this lower bridge appears near minimap row 52.

## Scientific status

The default is a **privileged state-oracle, joint-action-matched baseline**. It
reads exact raw unit coordinates, health, tags, and weapon cooldowns, then emits
one joint command with the benchmark branches: noop, one of eight 2-world-unit
compass moves, or an in-range targeted attack plus a unit-selection mask.
Navigation selects all living friendly tags—the RAW API equivalent of starting
with all Marines selected. Label the observation access as privileged when
reporting it. The optional `per_unit_focus_fire_kite` mode targets the reduced
per-unit interface instead and is identified separately in result metadata.

Do not compare its observation access directly with a learned actor whose
observation omits coordinates or cooldowns. Use it as a transparent physical
solvability/control reference and label it as an oracle in the paper.

The evaluator directly constructs one PySC2 environment because the current
legacy joint wrappers alias their x/y movement caches, while current per-unit
wrappers do not nest multiple same-frame PySC2 actions correctly. The script
keeps the intended public action semantics without relying on either defect.

## Run one variant

From the repository root:

```powershell
TBMsc2\Scripts\python.exe Agents\scripted\V1_Base\run_scripted.py --episodes 10
TBMsc2\Scripts\python.exe Agents\scripted\V2_Combat\run_scripted.py --episodes 10
TBMsc2\Scripts\python.exe Agents\scripted\V3_Navigate\run_scripted.py --episodes 10
```

Every variant folder contains the same thin launcher. To run all nine:

```powershell
TBMsc2\Scripts\python.exe Agents\scripted\run_all_variants.py --episodes 100
```

Useful options:

```text
--tactic focus_fire|focus_fire_kite|per_unit_focus_fire_kite  # default: focus_fire_kite
--seed 0
--save-replays
--visualize
--realtime
--map-dir <camera-free-map-directory>
--output-dir <results-root>
--run-name <stable-run-name>
--no-save-results
```

By default, results are written to a timestamped directory under
`Agent Performance Charts/Scripted/` as:

- `scripted_results.json` (metadata, aggregate results, and episode records);
- `scripted_episodes.csv` (one row per episode); and
- `scripted_summary.csv` (one row per map, including Wilson 95% intervals).

The episode file logs spawn regions, whether the lower bridge was reached,
first contact/attack/damage steps, kills, survivors, and the terminal outcome.
Combat and navigation wins are reported separately: an incidental beacon
capture is not counted as evidence of combat solvability.

PySC2 normally reuses one game seed on every reset. The evaluator advances the
SC2 seed for each episode (and records it in every row), giving deterministic
but non-identical trigger layouts.

For a reviewer-facing table, use enough episodes to cover randomized region
assignments rather than treating a one-episode smoke test as a success-rate
estimate. V3 is a 5-vs-8 disadvantage and should not be described as combat
solved unless the aggregate script results actually demonstrate wins.

## Static tests

```powershell
TBMsc2\Scripts\python.exe -m unittest Agents.scripted.tests.test_policy -v
```
