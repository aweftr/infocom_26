# PADMA: Prediction-Aware Dynamic VM Scheduling in Multi-NUMA Clouds via Theoretical Analysis and Transformer-based PPO

## Data
The VM dataset used in this paper is the [Huawei-East-1](https://github.com/huaweicloud/VM-placement-dataset/tree/main?tab=readme-ov-file) dataset from Huawei Cloud. The original dataset is available at [`data/Huawei-East-1.csv`](data/Huawei-East-1.csv). In our environment, VMs may experience delays, meaning their start times are not fixed. Therefore, we modify the dataset format to treat the VM's runtime (`lt`, length time) as an inherent property of the VM.

## Usage

- Install poetry 2.0+
- Run `poetry sync` to install python packages.(python >= 3.12, pytorch>=2.7)
- Run `poetry env activate` to activate the install python environment. 
- Run and play with the code!