# Flow control logic placement under failures

Data and code for the paper "A Study of Flow Control Logic Placement under
Failures in Data Center Networks".

The paper compares two places for the decision that reacts to a failure in a
data center network. A local decision point, such as a switch, reacts first
but sees only part of the network. A global one, a controller, sees the whole
network but reacts later. The paper has a survey of published mechanisms and
a model with a decision rule. This repository has

- the lists of papers that went through the survey;
- the scripts that print the survey numbers, check the model by Monte Carlo
  simulation and draw the figures.

## Survey data

All files are in `survey/`.

| File | Rows | Content |
|---|---|---|
| `keyword_corpus.csv` | 650 | records found by the keyword search in IEEE Xplore and ACM Digital Library on 2026-10-28, duplicates removed. Three of them are service entries (a table of contents, a title page, an untitled record) |
| `assessed_papers.csv` | 512 | records of the assessment stage with the verdict (`include` or `exclude`) and the number of passes |
| `survey_table.csv` | 177 | included mechanisms with the coded data used in the paper |
| `added_channels.csv` | 212 | records from OpenAlex and snowballing that passed the screening of titles and abstracts but were not assessed in full text. No number in the paper is computed from them |

Records are matched across the files by the `id` column.

Screening and assessment were done by an LLM classifier that was given the
criteria, and the verdicts are published as produced.

The files hold bibliographic data and coded fields only. Full texts,
abstracts and quotes from the papers are not included; every paper can be
found by its DOI or URL.

Main columns of `survey_table.csv`:

| Column | Meaning |
|---|---|
| `placement` | placement point: switch silicon, end host, software edge, distributed routing or central controller |
| `view` | what the decision point sees |
| `tau_c_seconds` | published reaction time in seconds, if the paper gives an absolute value |
| `tau_c_kind` | class of the published reaction-time figure: `loop_closure` and `detection_to_reroute` are reaction times, `periodic_interval` is a probing or reconfiguration period, `relative_only` is a gain without an absolute value, `other_quantity`, `none` |
| `tau_c_includes_detection` | whether detection is part of the figure |
| `tau_c_trigger` | what the figure is a reaction to: `failure`, `congestion` or `other` |
| `tau_c_measured_on` | testbed, simulation, hardware or analytical |
| `failure_as_process` | whether a failure is modelled as lasting a random time |
| `control_failure_considered` | whether failure of the control means is considered |
| `reports_tail_metric`, `reports_deadline_miss` | result of the keyword rule over the full text |
| `deadline_miss_share` | manual reading of the texts where the deadline rule fired |

## Scripts

Survey:

| Script | What it does |
|---|---|
| `survey/survey_stats.py` | prints every survey number used in the paper |
| `survey/survey_map.py` | draws the map "reaction time x view" (`figures/fig1.pdf`) |
| `survey/keyword_rule.py` | keyword search over the full texts for tail metrics and deadline misses. It needs the full texts, which are not in the repository; its result is already stored in `survey_table.csv` |

Model. The notation follows the paper: `m` is the mean failure duration,
`t_L` and `t_G` are the reaction times of the local and of the global point,
`r` is the share of affected flows that stay late after the local reaction,
and `k = (t_G - t_L)/m`. The rule says that the global point is preferable
when `k < ln(1/(1-r))`.

| Script | What it does |
|---|---|
| `model/sim.py` | Monte Carlo simulation of the model. It integrates the loss over the timeline of each failure and compares the mean with the closed form |
| `model/make_table.py` | LaTeX table "closed form versus simulation" |
| `model/tail_boundary.py` | boundary of the rule for non-exponential failure durations (`figures/fig2.pdf`) |
| `model/decision_map.py` | decision map in the plane (r, m) (`figures/fig3.pdf`) |
| `model/ext_a_boundary.py` | shift of the boundary when a decision point may be unavailable, and the loss with overlapping failures |
| `model/verify_sympy.py` | symbolic check of the formulas |
| `model/illustration.py` | the rule applied to published failure durations |

## Running

```bash
pip install -r requirements.txt

python3 survey/survey_stats.py
python3 survey/survey_map.py

python3 model/verify_sympy.py
python3 model/sim.py --runs 1000000 --seed 42
python3 model/sim.py --scenario grid --out model/grid.csv
python3 model/make_table.py --runs 1000000 > model/table_sim.tex
python3 model/tail_boundary.py
python3 model/decision_map.py
python3 model/ext_a_boundary.py
python3 model/illustration.py
```

Run the scripts from the repository root. The figures go to `figures/`.

All seeds are fixed. The results were obtained with Python 3.12, numpy 2.2,
scipy 1.15, matplotlib 3.10 and sympy 1.14. With these settings the expected
losses printed by `make_table.py` are within 0.12 % of the closed forms, the
rule is reproduced at 568 of the 570 grid points, and all 28 symbolic checks
pass.
