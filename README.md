# Inverse-designed spinodoid metamaterials
Implementation of machine learning framework for "Inverse-designed Spinodoid Metamaterials" (https://doi.org/10.1038/s41524-020-0341-6) as described in the following publication.

## Citation
If you use this code, please cite the following publication:
S. Kumar, S. Tan, L. Zheng, D.M. Kochmann, **Inverse-designed spinodoid metamaterials**, _npj Comput Mater 6, 73_ (2020). https://doi.org/10.1038/s41524-020-0341-6

Note: To generate the spinodoid designs (predicted from machine learning), see: https://www.gibboncode.org/html/HELP_spinodoid.html.

## Requirements

- Python (tested on version 3.7.1)
- Python packages:
    - PyTorch (tested without CUDA)
    - NumPy
    - pandas
    - statistics

## Usage

```sh
python main.py
```

## File descriptions
- main.py: main file to be executed and contains training protocols
- model.py: functions for creating neural network models
- loadDataset.py: functions for loading data from data.csv
- errorAnalysis.py: functions for post-processing and error analysis
- normalization.py: functions for normalization of features (inputs to neural networks)
- parameters.py: contains all parameters and hyper-parameters for neural network architectures and training protocols

## Outputs
After training is over, outputs will be available in the following directories:
- ./models/ : contains trained models
- ./loss-history/ : contains loss history during training

