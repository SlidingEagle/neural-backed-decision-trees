
# Neural-Backed Decision Trees

NBDT match or outperform modern neural networks on CIFAR10, CIFAR100, TinyImagenet200.
To reproduce experimental results, clone the repository, install all requirements, and run our bash script.

```bash
cd neural-backed-decision-trees
python setup.py develop # install all requirements
bash scripts/gen_train_eval_wideresnet_cifar10.sh # reproduce paper core CIFAR10 results
bash scripts/gen_train_eval_wideresnet_cifar100.sh # reproduce paper core CIFAR100 results
bash scripts/gen_train_eval_wideresnet_tiny.sh # reproduce paper core TinyImagenet200 results
```
