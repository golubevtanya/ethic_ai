## About The Project
This repository contains a notebook and related scripts with approaches to debais [Adult census](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset) dataset.
The notebook contains explanations and graphs.

### Debaising approaches 
* Blinding (delete sensetive attributes) 
* Balancing amount of data for sensetive attributes
* Equalized Odds Postprocessing

  
The The  technique was done using [AI Fairness 360](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.algorithms.postprocessing.EqOddsPostprocessing.html#aif360.algorithms.postprocessing.EqOddsPostprocessing)).


Example of using Calibrated Equalized Odds Postprocessing on the Adult dataset can be found [here](https://github.com/Trusted-AI/AIF360/blob/main/examples/demo_calibrated_eqodds_postprocessing.ipynb).

IBM copyright: Copyright 2018 - 2024, The AI Fairness 360 (AIF360) Authors. Revision 7c4f172f.

