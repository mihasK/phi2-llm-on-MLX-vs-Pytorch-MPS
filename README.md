# Benchmark of running Phi-2 LLM on MLX and PyTorch on MPS device

Code for the [article](https://medium.com/@koypish/the-tools-you-a-python-developer-need-on-your-new-mac-68d8d1ebbff) about speed comparison of MLX and PyTorch on Apple's M1 MAX GPU: https://medium.com/@koypish/the-tools-you-a-python-developer-need-on-your-new-mac-68d8d1ebbff

## Installation
* `pyenv install 3.11`
* `pyenv local 3.11`
* `poetry shell`

## Run Phi-2
* `make mlx`
* `make mps` - for PyTorch on Metal GPU
* `make cpu` - for PyTorch on CPU

In order to track CPU/GPU usage, keep `make track` running while performing operation of interest.

## Structure
* `utils.py` contains memory measurement utils
* `try_phi2_torch.ipynb`, `try_phi2_mlx.ipynb` - benchmarks code
* `phi2_mlx.py` - code mainly copied from https://github.com/ml-explore/mlx-examples/tree/main/llms/phi2
* `plot_metrics.ipynb` - contains code to parse and analyze output of `powermetrics` (run by `make track`) 
  

