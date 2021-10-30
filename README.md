# Prompt-tuning for Compositional Generalization

## Setup local development
### Install dependencies
1. Create a Conda env:
```bash
conda create -n grokking python=3.8
conda activate grokking
``` 
2. Install torch
3. Install the rest of dependencies:
```bash
pip install -r src/requirement.txt
```
3. Configure Comet ML
```bash
ipython
```
Run the following:
```python
import comet_ml
comet_ml.init()
```
### Download some dataset
```bash
python scripts/download_ds.py scan-simple_debug
```
### Run the model
Train and eval
```bash
python src/main.py --configs "configs/power_et_al2.conf,configs/data/moddiv.conf" train
```

Check out the `experiments` directory.

## Mila Cluster
### Setup environment
1. Clone the repo
```bash
git clone https://github.com/kazemnejad/grokking.git
cd grokking
```
2. Create a link to `launch_experiment.py`
```bash
chmod a+x $PWD/scripts/launch_experiment.py
ln -sf $PWD/scripts/launch_experiment.py ~/.local/bin/gr_launch_exp
```
3. Setup a conda env for `launch_experiment.py`
```bash
module load miniconda/3
conda create -n grokking python=3.8
conda activate grokking
pip install comet_ml ipython
```
4. Setup Comet ML:
```bash
ipython
```
```python
import comet_ml
comet_ml.init()
```
5. Download the Singularity image
```bash
module load singularity
mkdir -p ~/scratch/containers
cd ~/scratch/containers
singularity pull library://amrhssn/default/image:latest
mv image_latest.sif pt.sif
```

### Launch a job
We use the following command:
```
usage: launch_experiment.py [-h] [-p PLATFORM] [-s ARGS] [-c ARGS] [-a ASSETS] [-w WORKSHEET] [-i IMAGE] [--images-dir DIR] [--lt-storage DIR] [--node-storage DIR] [--account ACCOUNT] [--script-dir DIR] [--log-dir DIR] [--env ENVS] [--interactive] [--tb-on-interactive]
                            [--install-notify] [--notify-webhook-key NOTIFY_WEBHOOK_KEY] [--notify-event-name NOTIFY_EVENT_NAME] [--list] [-n N] [--upload] [--download]
                            [BUNDLE_ID]

Experiment runner

positional arguments:
  EXP_KEY             Experiment Key

optional arguments:
  -h, --help            show this help message and exit
  -p PLATFORM, --platform PLATFORM
                        The computation platform we're running the experiment
  -s ARGS, --slurm-args ARGS
                        Slurm args
  -c ARGS, --cl-args ARGS
                        Codalab `run` args
  -a ASSETS, --assets ASSETS
                        Experiment assets that should be copied to container
  -i IMAGE, --image IMAGE
                        Container Image
  --env ENVS            Environment variables passed to the container, e.g. X1=V1,x2=V2
  --interactive         Run in the interactive mode
  --list                List all experiments on this platform
  -n N                  Number of items in the list
```

Here is an example:
```bash
module load miniconda/3
conda activate grokking
gr_launch_exp --slurm-args "--gres=gpu:48gb:1 --partition=long -t 3:00:00 -c 4 --mem=10G" \
    --image pt.sif <exp_key>
```

### Upload an experiment
We use the following script:
```
usage: upload_experiment.py [-h] [-s CONFIGS[,CONFIGS,CONFIGS]] [-c cmd -a -b[,cmd -c -d]] [-d DATASET] [-p project] [-e KEY=VAL[,KEY=VAL]]

Make Experiment Bundle

optional arguments:
  -h, --help            show this help message and exit
  -s CONFIGS[,CONFIGS,CONFIGS], --configs CONFIGS[,CONFIGS,CONFIGS]
                        Config file names
  -c cmd -a -b[,cmd -c -d], --commands cmd -a -b[,cmd -c -d]
                        Experiment commands
  -d DATASET, --dataset DATASET
                        Dataset name's bundle
  -p project, --project project
                        CometML project
  -e KEY=VAL[,KEY=VAL], --env-vars KEY=VAL[,KEY=VAL]
                        Experiment environment variables
```
Here is an example:
```bash
python scripts/upload_experiment.py --configs "configs/fine_tune_debug.conf,configs/data/scan.conf" \
    --commands "train"
```
