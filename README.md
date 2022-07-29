# pseudoESM


## Launch MLFlow

### Set Environment Variables
```
conda env config vars set MLFLOW_TRACKING_URI=http://127.0.0.1:8889
conda env config vars set MLFLOW_EXPERIMENT_NAME=pseudoESM
conda env config vars set REGISTERED_MODEL_NAME=pseudoESM
```

```bash
mlflow server --default-artifact-root file://./mlruns --host 0.0.0.0 --port 8889
```


## Training
```bash

## GPU Training
./distribute_train.sh
```

```bash
## modify configs/train.yaml
## - mode.gpu = false
## - train.amp = false
./train.sh
```


## Inference

```bash
python inference.py
```

## distribute problem solve
```bash
1、you better use conda to install 'mpi' and 'nccl'
2、If your GPU usage reaches 100%，there must be something deadlocked. First, check init_process_group method and use 'gloo'
```