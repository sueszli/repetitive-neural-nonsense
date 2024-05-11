goal: implement a news recommendation algorithm, that predicts user engagement with articles.

assigned model: gru - gated recurrent unit

_general competition links:_

-   assignment: https://tuwel.tuwien.ac.at/mod/page/view.php?id=2281418
    -   clarification: https://tuwel.tuwien.ac.at/mod/moodleoverflow/discussion.php?d=8864
-   gitlab: https://gitlab.tuwien.ac.at/recsys-laboratory/teaching/24ss-recsys-lecture/Group_26
-   acm recsys challenge: https://www.recsyschallenge.com/2024/
-   acm conference: https://recsys.acm.org/recsys24/challenge/
-   codabench: https://www.codabench.org/
    -   competition: https://www.codabench.org/competitions/2469/
    -   our team: https://www.codabench.org/profiles/organization/264/edit/
    -   faq: https://www.codabench.org/forums/2387/

_development links:_

-   dataset:

    -   description: https://recsys.eb.dk/dataset/

-   ebnerd-benchmark:

    -   quickstart: https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/nrms_ebnerd.ipynb
    -   LSTUR implementation: https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/lstur_dummy.py 👈

-   recommenders:
    -   quickstart: https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/00_quick_start/dataset_ebnerd.ipynb
    -   LSTUR implementation: https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/lstur_MIND.ipynb 👈
    -   code: https://github.com/recommenders-team/recommenders/blob/main/recommenders/models/deeprec/models/sequential/gru.py
    -   docs: https://recommenders-team.github.io/recommenders/models.html#gru

_rnn theory:_

-   https://colah.github.io/posts/2015-08-Understanding-LSTMs/
-   https://colah.github.io/posts/2015-09-NN-Types-FP/
-   https://karpathy.github.io/2015/05/21/rnn-effectiveness/
-   statquest: https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1

_steps:_

-   register group and algorithm on tuwel, register organization on "codabench" ✅
-   download dataset ✅

-   implement 3 algorithms:

    -   baseline model from the repository "NRMS on EB-NeRD" from "ebnerd-benchmark" ✅
    -   "GRU" from "recommenders" repository (it's the core of LSTUR) ✅
    -   some algorithm of choice

-   improve: check out "beyond metrics" section in the "ebnerd-benchmark" repository
-   write report pdf
-   submit to "codabench"
-   submit to gitlab, add "final" tag
-   upload project report pdf
