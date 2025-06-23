### This is a repo for basic ImageNet1K running validation eval. Additionally, also to train a classifier on top of frozen ImageNet1k features on downstream datasets - CIFAR100.

**Note**: `cifar100_train_eval_linear_classifier_gridsearch_buggy.py` is buggy with `params.requires_grad=False` but `BN` is in `.train()` mode when the `model.train()` step takes place. `cifar100_train_eval_linear_classifier_gridsearch.py` is the fixed file.
