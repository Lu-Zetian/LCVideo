# LCVideo

### Set Up

1. Create a new environment (recommended but not required)

```
conda create -n lcvideo python=3.11
```

2. Install the library
```
pip install -r requirements.txt
```

3. Install pretrainded model
```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 pretrains/SD/
```

### Inference
1. Update `configs/inference/inference_config.yaml`
2. Run generate_video.py
```
python generate_video.py --config path/to/config.yaml
```

### Training
1. Update `configs/inference/training_config.yaml`
2. Run train.py
```
python train.py --config path/to/config.yaml
```