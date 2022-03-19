# JointFormer

## Trained weights

Trained weights for the Jointformer on CPN keypoints can be found [here](https://drive.google.com/file/d/1NEgtjHmgS8YZtRX2MObLEo0oRgSDS5Tg/view?usp=sharing). The parameters we used to train these weights were:
```
python run_jointformer.py --batch_size 256 --num_workers 2 --epochs 50 --keypoints cpn_ft_h36m_dbb --hid_dim 64 --intermediate --pred_dropout 0.2 --augment
```

Trained weights for the RefinementTransformer on CPN keypoints use the trained weights of the JointFormer for the first stage. Our trained weights can be found [here](https://drive.google.com/file/d/1KY0Cxb5mo6woqP0bIOOmfHaAvpzmM1h3/view?usp=sharing). The parameters we used to train these weights were:
```
python run_refinement.py --batch_size 256 --num_workers 2 --epochs 30 --keypoints cpn_ft_h36m_dbb --hid_dim 256 --num_layers 2 --pred_dropout 0.1 --augment --d_inner 1024
```

Trained weights for the Jointformer on GT keypoints can be found [here](https://drive.google.com/file/d/10OosSADlT2gyh68D-zPJpovEkXqv1EeL/view?usp=sharing). The parameters we used to train these weights were:
```
python run_jointformer.py --batch_size 256 --num_workers 2 --epochs 50 --keypoints gt --hid_dim 64 --intermediate --pred_dropout 0.2 --augment  
```
