# MultiPharmPre
# Multimodal Pharmacophore Pre-training for Boosting Generalization Capability in Molecular Property Prediction

# 1. MultiPharmPre setup
Dependencies:
- python 3.7
- pytorch = 1.7.1
- torch-cluster = 1.5.9
- torch-geometric = 1.7.2
- torch-scatter = 2.0.7
- torch-sparse = 0.6.9
- torch-spline-conv = 1.2.1
- RDkit = 2021.03.3
- numpy
- pandas

# 2. MultiPharmPre Pre-training
   python MultiPharmPre_pretrain.py --epochs 1000 --dataset zinc --model RGNN_pretrining --cuda 0 --checkpoint

# 3. MultiPharmPre Fine-tuning
   python MultiPharmPre_Classification.py --epochs 200  --dataset hERG --model RGNN_Classification --cuda 0  --checkpoint --pretrain RGNN_best_model.ckp

# 4. MultiPharmPre preditcion
   use calculate_acc.py to analysis the prediction results
