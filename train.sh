# python train.py --config configs/maml/ant-soft-metaRL.yaml --output-folder ant-soft-metaRL --seed 1 --num-workers 2

# python train_ray.py --config configs/maml/ant-soft-metaRL-ray.yaml --output-folder ant-box-weight-ray-0.5-1 --seed 1
# python train_ray.py --config configs/maml/ant-soft-metaRL-ray.yaml --output-folder ant-box-weight-ray-0.5-2 --seed 2
# python train_ray.py --config configs/maml/ant-soft-metaRL-ray.yaml --output-folder ant-box-weight-ray-0.5-3 --seed 3


# python train_ray.py --config configs/maml/halfcheetah-soft-metaRL-ray.yaml --output-folder halfcheetah-leg-weight-ray-0.1 --seed 1

# python train_ray.py --config configs/maml/snapbot-4leg-ray.yaml --output-folder snapbot-4leg-ray-0.05 --seed 1

python train_ray.py --config configs/maml/snapbot-6leg-ray.yaml --output-folder snapbot-6leg-ray-0.05 --seed 1
