# Reduced-agent comparisons with scripted oracle

These are new comparison figures; the original learned-agent plots under
`Reduced Agent Aggregate Plots/` are not modified.

The checkpoint win-rate grid adds the scripted agent as a fixed dashed
horizontal reference in every map panel. It is not assigned fabricated
training checkpoints. The light band is the Wilson 95% confidence interval
for its 32 evaluation episodes.

The terminal-outcome grid adds a fourth row for the scripted oracle. Learned
rows pool the final eligible checkpoints from three training seeds (96
episodes per map); the scripted row contains one fixed-policy evaluation of 32
episodes per map. The source CSV records those different sample sizes and the
input paths.

Regenerate the current 10M/8-evaluation-environment comparison from the
repository root with:

```powershell
TBMsc2\Scripts\python.exe "Agent Performance Charts\reduced_agents_with_scripted.py" --grid 10m
```

The plotted scripted baseline reads privileged raw state and should be labeled
as a scripted oracle, not as an observation-matched learned policy.
