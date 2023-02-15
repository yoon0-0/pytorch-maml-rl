# python test.py --config mp/config.json --policy mp/policy.th --output mp/result.npz --num-workers 2
# python test_ray.py --config ant-soft-metaRL-ray-1//config.json --policy ant-soft-metaRL-ray-1/policy.th499 --output ant-soft-metaRL-ray-1/result.npz --seed 2


# python test_ray.py --config ant-box-weight-ray-0.5-1/config.json --policy ant-box-weight-ray-0.5-1/policy.th499 --output ant-box-weight-ray-0.5-1/result.npz --seed 1
# python test_ray.py --config ant-box-weight-ray-0.5-2/config.json --policy ant-box-weight-ray-0.5-2/policy.th499 --output ant-box-weight-ray-0.5-2/result.npz --seed 2
# python test_ray.py --config ant-box-weight-ray-0.5-3/config.json --policy ant-box-weight-ray-0.5-3/policy.th499 --output ant-box-weight-ray-0.5-3/result.npz --seed 3

# python test_ray.py --config ant-leg-weight-ray-0.5-1/config.json --policy ant-leg-weight-ray-0.5-1/policy.th499 --output ant-leg-weight-ray-0.5-1/result.npz --seed 1
# python test_ray.py --config ant-leg-weight-ray-0.5-2/config.json --policy ant-leg-weight-ray-0.5-2/policy.th499 --output ant-leg-weight-ray-0.5-2/result.npz --seed 2
# python test_ray.py --config ant-leg-weight-ray-0.5-3/config.json --policy ant-leg-weight-ray-0.5-3/policy.th499 --output ant-leg-weight-ray-0.5-3/result.npz --seed 3

python test_ray.py --config snapbot-6leg-ray-0.05/config.json --policy snapbot-6leg-ray-0.05/policy.th1 --output snapbot-6leg-ray-0.05/4leg_transfer/result.npz --seed 1