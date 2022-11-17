# change-that-matters-ACL2022

This is the code repository for [The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317) to appear in Findings of [ACL 2022](https://www.2022.aclweb.org). 

This research was conducted in conjunction with [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://openreview.net/pdf?id=S0lx6I8j9xq) to appear in [UAI 2022](https://www.auai.org/uai2022/).

Some of the code is shared across repositories as detailed below.

## Shared Code
Code for summarizing raw results (e.g., producing regression plots) is available in this repository. Additional code for summarizing results as well as code for running all experiments is available in the shared repository [here](https://github.com/anthonysicilia/multiclass-domain-divergence). Further, a python package designed to compute the statistic we propose is available [here](https://github.com/anthonysicilia/classifier-divergence).

Please consider citing one or both papers if you use this code.

## Relevant Links
arXiv (ACL 2022): https://arxiv.org/abs/2203.11317

OpenReview (UAI 2022): https://openreview.net/pdf?id=S0lx6I8j9xq

shared code: https://github.com/anthonysicilia/multiclass-domain-divergence

UAI code: https://github.com/anthonysicilia/pacbayes-adaptation-UAI2022

package: https://github.com/anthonysicilia/classifier-divergence

### Notable Versions
Code was run using the following versions (some packages are only used by shared repos):
 - python==3.7.4
 - matplotlib==3.5.0
 - numpy==1.21.2
 - pandas==1.3.5
 - scipy==1.7.3
 - seaborn==0.12.1
 - torch==1.10.2 (build py3.7_cuda10.2_cudnn7.6.5_0)
 - tqdm==4.45.0
 - pillow==8.4.0
 - statsmodels==0.13.0
