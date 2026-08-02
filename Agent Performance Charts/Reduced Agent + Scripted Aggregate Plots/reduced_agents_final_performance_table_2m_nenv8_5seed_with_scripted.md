# Final agent performance - 2M, 8 eval envs, 5 training seeds

| Map variant | Agent | Final mean win rate [seed range] | Nav win (%) | Combat win (%) | Combat loss (%) | Timeout loss (%) |
|---|---|---:|---:|---:|---:|---:|
| `V1_Base` | QMIX | 10.0 [0.0, 34.4] | 10.0 | 0.0 | 0.0 | 90.0 |
| `V1_Base` | **MaskPPO** | **25.6 [3.1, 43.8]** | 25.6 | 0.0 | 0.0 | 74.4 |
| `V1_Base` | MAPPO | 11.2 [0.0, 43.8] | 11.2 | 0.0 | 0.0 | 88.8 |
| `V1_Base` | Scripted oracle | 96.9 [N/A, N/A] | 0.0 | 96.9 | 3.1 | 0.0 |
| `V1_Combat` | QMIX | 8.1 [3.1, 12.5] | 7.5 | 0.6 | 11.2 | 80.6 |
| `V1_Combat` | MaskPPO | 13.8 [3.1, 28.1] | 13.8 | 0.0 | 0.0 | 86.2 |
| `V1_Combat` | **MAPPO** | **19.4 [0.0, 53.1]** | 19.4 | 0.0 | 13.8 | 66.9 |
| `V1_Combat` | Scripted oracle | 100.0 [N/A, N/A] | 0.0 | 100.0 | 0.0 | 0.0 |
| `V1_Navigate` | QMIX | 17.5 [0.0, 46.9] | 17.5 | 0.0 | 0.0 | 82.5 |
| `V1_Navigate` | MaskPPO | 52.5 [28.1, 100.0] | 52.5 | 0.0 | 0.0 | 47.5 |
| `V1_Navigate` | **MAPPO** | **78.1 [46.9, 100.0]** | 78.1 | 0.0 | 0.0 | 21.9 |
| `V1_Navigate` | Scripted oracle | 100.0 [N/A, N/A] | 0.0 | 100.0 | 0.0 | 0.0 |
| `V2_Base` | QMIX | 0.6 [0.0, 3.1] | 0.6 | 0.0 | 0.0 | 99.4 |
| `V2_Base` | **MaskPPO** | **25.6 [12.5, 34.4]** | 25.6 | 0.0 | 0.0 | 74.4 |
| `V2_Base` | MAPPO | 3.8 [0.0, 12.5] | 3.8 | 0.0 | 6.9 | 89.4 |
| `V2_Base` | Scripted oracle | 37.5 [N/A, N/A] | 0.0 | 37.5 | 62.5 | 0.0 |
| `V2_Combat` | QMIX | 3.8 [0.0, 9.4] | 3.8 | 0.0 | 3.8 | 92.5 |
| `V2_Combat` | MaskPPO | 16.9 [0.0, 53.1] | 16.9 | 0.0 | 0.0 | 83.1 |
| `V2_Combat` | **MAPPO** | **20.6 [0.0, 53.1]** | 20.6 | 0.0 | 0.0 | 79.4 |
| `V2_Combat` | Scripted oracle | 25.0 [N/A, N/A] | 0.0 | 25.0 | 75.0 | 0.0 |
| `V2_Navigate` | QMIX | 11.9 [0.0, 37.5] | 11.9 | 0.0 | 0.0 | 88.1 |
| `V2_Navigate` | MaskPPO | 34.4 [0.0, 78.1] | 34.4 | 0.0 | 0.0 | 65.6 |
| `V2_Navigate` | **MAPPO** | **58.1 [40.6, 75.0]** | 58.1 | 0.0 | 0.0 | 41.9 |
| `V2_Navigate` | Scripted oracle | 28.1 [N/A, N/A] | 0.0 | 28.1 | 71.9 | 0.0 |
| `V3_Base` | QMIX | 13.1 [0.0, 28.1] | 13.1 | 0.0 | 0.0 | 86.9 |
| `V3_Base` | **MaskPPO** | **13.8 [6.2, 25.0]** | 13.8 | 0.0 | 0.0 | 86.2 |
| `V3_Base` | MAPPO | 4.4 [0.0, 9.4] | 4.4 | 0.0 | 0.0 | 95.6 |
| `V3_Base` | Scripted oracle | 0.0 [N/A, N/A] | 0.0 | 0.0 | 100.0 | 0.0 |
| `V3_Combat` | QMIX | 4.4 [0.0, 12.5] | 4.4 | 0.0 | 10.6 | 85.0 |
| `V3_Combat` | **MaskPPO** | **11.2 [6.2, 15.6]** | 11.2 | 0.0 | 0.0 | 88.8 |
| `V3_Combat` | MAPPO | 4.4 [0.0, 9.4] | 4.4 | 0.0 | 3.1 | 92.5 |
| `V3_Combat` | Scripted oracle | 0.0 [N/A, N/A] | 0.0 | 0.0 | 100.0 | 0.0 |
| `V3_Navigate` | QMIX | 16.2 [0.0, 40.6] | 16.2 | 0.0 | 0.0 | 83.8 |
| `V3_Navigate` | MaskPPO | 37.5 [0.0, 90.6] | 37.5 | 0.0 | 0.0 | 62.5 |
| `V3_Navigate` | **MAPPO** | **76.2 [46.9, 96.9]** | 76.2 | 0.0 | 0.0 | 23.8 |
| `V3_Navigate` | Scripted oracle | 0.0 [N/A, N/A] | 0.0 | 0.0 | 100.0 | 0.0 |

Learned-agent win rates are the mean [minimum, maximum] across training seeds at the final checkpoint. Terminal outcomes are pooled across the same seed evaluations.

Each learned row uses 5 training seeds with 32 evaluation episodes per seed. The scripted oracle is one fixed 32-episode evaluation and has no training-seed range (N/A). All reported values are percentages. Bold marks the best learned agent per map; the privileged scripted oracle is excluded from that ranking.

Win rate is Nav win + Combat win. Reaching a terminal Combat loss is not counted as a win.
