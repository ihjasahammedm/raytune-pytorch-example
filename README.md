## Raytune-pytorch-example
This is a Python code repository that demonstrates the process of hyperparameter tuning for a Mask-RCNN model trained on a pedestrian dataset using Ray Tune. Hyperparameter tuning is a crucial step in machine learning model development, as it helps to identify the optimal set of hyperparameters that can lead to better model performance. 

#### Setup

- Download and unzip [dataset](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) to root directory before running the training.
- Run `pip install -r requirements.txt` to install dependencies. Python version >=3.7 is recommended.

#### Usage

`train.py`: This file contains the training function for the machine learning model. It takes a configuration dictionary as input, which specifies the hyperparameters to use for training the model. To train the model with
out tuning just run `python train.py` after setting the config parameters.

`tune.py`: This file contains the code for setting up and running the hyperparameter tuning process using Ray Tune. It defines a search space for the hyperparameters to tune, as well as a scheduler and search algorithm for the tuning process. It also specifies the number of samples to generate in the search space and the metric to optimize for.

The hyperparameters being tuned in this example are the learning rate and batch size, with the target metric being to achieve higher bbox mAP. The search space for the learning rate is defined using the tune.qloguniform function, which samples in log space and rounds to multiples of 0.0001. The batch size is being tuned using tune.choice, which samples from a list of possible values.

The scheduler being used is ASHA (Async Successive Halving), which is a popular algorithm for hyperparameter tuning. It works by running multiple trials in parallel and eliminating the lowest-performing trials at each iteration, based on their performance relative to the other trials. The search algorithm being used is HyperOptSearch, which is a search algorithm based on the Bayesian optimization approach.

The hyperparameter tuning process is performed by generating a set of trials with different hyperparameter values, training the model using each set of hyperparameters, and evaluating the model performance using the bbox mAP metric. The results of the hyperparameter tuning process are printed to the console at the end of the script, including the best trial configuration and the final metric value achieved.

The best result obtained after hyperparameter tuning is 0.5% higher than the initial value with the same number of epochs, suggesting that hyperparameter tuning can lead to improved model performance even when the initial parameters are already optimized. The summary of trials performed by Ray Tune provides insight into the search space explored during the tuning process, helping to identify the best hyperparameters for the model.
Best config and ray tune summary is shown below:
![image](/resources/best_result.jpg)
![image](/resources/summary.PNG)
Initial params and best params obtained after tuning are highlighted in summary.
