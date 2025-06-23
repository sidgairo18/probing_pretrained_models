python cifar100_train_eval_linear_classifier_gridsearch.py \
  --arch resnet50 \
  --batch_size 128 \
  --epochs 10 \
  --lr_list 0.1 0.01 0.001 0.0001 0.00001
