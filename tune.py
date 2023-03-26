import numpy as np
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from train import train, config

# in this example we are tuning lr and batchsize
search_space = {
    "lr": tune.qloguniform(1e-4, 1e-1, 1e-4),   # sampling in log space and rounding to multiples of 0.00001
    'batch_size' : tune.choice([2, 4])
}
# scheduler used is Async Successive Halving
# maximum iteration is set as 10 and all trials will be run atleast for 4 iteration
scheduler = ASHAScheduler(
    max_t=10,
    grace_period=4,
    reduction_factor=2)

initial_value = {'lr' : 0.005,
                 'batch_size' : 2}
search_alg = HyperOptSearch(points_to_evaluate=[initial_value])

def trainable(search_space, config):
    for key, value in search_space.items():
        config[key] = value
    config['raytune'] = True
    train(config)

tuner = tune.Tuner(
    tune.with_resources(tune.with_parameters(trainable, config=config),
                        resources={"cpu": 4, "gpu": 1}) # 4 core and 1 GPU allocated for each trial
    ,
    param_space=search_space,
    tune_config=tune.TuneConfig(metric="mAP", mode="max",
                                search_alg=search_alg,
                                scheduler=scheduler,
                                num_samples=20)
)
results = tuner.fit()
best_result = results.get_best_result("mAP", "max")

print("Best trial config: {}".format(best_result.config))
print("Best trial final mAP: {}".format(
    best_result.metrics["mAP"]))
