# JointFormer

This is the official repository of our paper [Jointformer: Single-Frame Lifting Transformer with Error Prediction and Refinement for 3D Human Pose Estimation](https://arxiv.org/abs/2208.03704) published at ICPR2022. This repository is based on the [SemGCN](https://github.com/garyzhao/SemGCN) repository. Please refer to their readme or to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to setup the training data.

## Trained weights

Trained weights for the Jointformer on CPN keypoints can be found [here](https://drive.google.com/file/d/1NEgtjHmgS8YZtRX2MObLEo0oRgSDS5Tg/view?usp=sharing). The parameters we used to train these weights were:
```
python run_jointformer.py --batch_size 256 --num_workers 2 --epochs 50 --keypoints cpn_ft_h36m_dbb --hid_dim 64 --intermediate --pred_dropout 0.2 --augment
```

Trained weights for the RefinementTransformer on CPN keypoints use the trained weights of the JointFormer for the first stage. Our trained weights can be found [here](https://drive.google.com/file/d/1KY0Cxb5mo6woqP0bIOOmfHaAvpzmM1h3/view?usp=sharing). The parameters we used to train these weights were:
```
python run_refinement.py --batch_size 256 --num_workers 2 --epochs 30 --keypoints cpn_ft_h36m_dbb --hid_dim 256 --num_layers 2 --pred_dropout 0.1 --augment --d_inner 1024 --pose_weights {path/to/pretrained/jointformer/weights}
```

Trained weights for the Jointformer on GT keypoints can be found [here](https://drive.google.com/file/d/10OosSADlT2gyh68D-zPJpovEkXqv1EeL/view?usp=sharing). The parameters we used to train these weights were:
```
python run_jointformer.py --batch_size 256 --num_workers 2 --epochs 50 --keypoints gt --hid_dim 64 --intermediate --pred_dropout 0.2 --augment  
```

To evaluate our trained weights and generate the evaluation values in the tables of the paper, please append `--evaluate {path/to/weights}` to the respective training commands.

## Bibtex
```
@article{lutz2021joint,
  title={Jointformer: Single-Frame Lifting Transformer with Error Prediction and Refinement for 3D Human Pose Estimation},
  author={Lutz, Sebastian and Blythman, Richard and Ghostal, Koustav and Moynihan Matthew and Simms, Ciaran and Smolic, Aljosa}
  journal={26TH International Conference on Pattern Recognition, {ICPR} 2022},
  year={2022}
}
```

## Acknowledgements
This repository is based on [SemGCN](https://github.com/garyzhao/SemGCN) and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). We thank their authors for releasing their code.
