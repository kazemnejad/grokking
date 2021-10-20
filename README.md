# Grokking

## How to use
### Install dependencies 
1. Install torch from its website
2. Install the rest:
```bash
conda create -n grokking python=3.8
conda activate grokking
conda install pytorch torchvision torchaudio -c pytorch
pip install src/requirement.txt
```
3. Configure Comet ML (Optional)
```bash
ipython
```
Run the following:
```python
import comet_ml
comet_ml.init()
```

### Train and Evaluation
```bash
python src/main.py --configs "configs/power_sum.conf" train
```