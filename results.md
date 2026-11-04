## Policy success on unseen tub/notebook layouts

| Policy | Training data | Episodes | 100 layouts | 300 layouts (95% CI) |
|---|---|---|---|---|
| ACT | your demos only, original spot | 17 | 6/100 | **5%** (3-8%) |
| ACT | scripted motions, NO human data | 1107 | 81/100 | **79%** (74-83%) |
| ACT | scripted motions, NO human data (same size as yours) | 553 | - | **80%** (75-84%) |
| ACT | your demos + augmentation | 587 | 86/100 | **81%** (76-85%) |
| ACT (seed 2) | your demos + augmentation | 587 | - | **82%** (78-86%) |
| ACT-JEPA | your demos + augmentation | 587 | 86/100 | **82%** (78-86%) |
| ACT-JEPA (seed 2) | your demos + augmentation | 587 | - | **85%** (81-89%) |

## World-model probe (frozen encoder -> future joint states, held-out episodes)

| Policy | RMSE (rad) | ATE (rad) |
|---|---|---|
| ACT | 0.0651 | 0.1458 |
| ACT (seed 2) | 0.0603 | 0.1492 |
| ACT-JEPA | 0.0653 | 0.1455 |
| ACT-JEPA (seed 2) | 0.0568 | 0.1385 |
