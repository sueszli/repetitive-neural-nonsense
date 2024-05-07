assignment: https://tuwel.tuwien.ac.at/mod/page/view.php?id=2281418

gitlab: https://gitlab.tuwien.ac.at/recsys-laboratory/teaching/24ss-recsys-lecture/Group_26

model: gru

goal: use dataset to build a news recommendation algorithm, that predicts user engagement with articles

---

links:

-   acm recsys challenge: https://www.recsyschallenge.com/2024/

---

group name: Group 26+

steps:

-   contact the lecturers and notify them about the group
-   register on evaluation platform with group name: https://www.codabench.org/competitions/2469/?secret_key=98314b2c-9237-471e-905c-2a88bf6a1d8a
-   download dataset / run baseline:
    -   EB-NeRD dataset / baseline: https://recsys.eb.dk/dataset/
    -   https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/nrms_ebnerd.ipynb
-   implement best practice model:
    -   docs: https://recommenders-team.github.io/recommenders/models.html#gru
    -   code: https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/deeprec/models/sequential/gru.py
-   implement your own model
