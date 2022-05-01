# Q2A Encoder

1. Follow [Q2A/README.md](https://github.com/showlab/Q2A/blob/master/README.md) to install required libraries and dataset.

2. Revise the dataset road of [Q2A/encoder/configs/vit_b16_384_fps1_train.yaml](https://github.com/showlab/Q2A/blob/master/encoder/configs/vit_b16_384_fps1_train.yaml).

3. Run the script:

```
cd Q2A/encoder
```

For video encoding, 

```
python main.py --cfg configs/vit_b16_384_fps1_train.yaml FOR.VIDEO True
```

For script encoding, 

```
python main.py --cfg configs/vit_b16_384_fps1_train.yaml FOR.SCRIPT True
```

For QA encoding,

```
python main.py --cfg configs/vit_b16_384_fps1_train.yaml FOR.QA True
```
