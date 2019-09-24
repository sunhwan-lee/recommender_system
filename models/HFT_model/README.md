## Hidden Factors as Topics (HFT) model

This repository contains code for the Recsys 2013 paper *[Hidden factors and hidden topics: understanding rating dimensions with review text](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf)*. For an intuitive overview of the paper, read the [blog post](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html).

## Dataset
Dataset used in the paper and many others can be found in [here](https://cseweb.ucsd.edu/~jmcauley/datasets.html) from the repository of prof. [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) in UCSD.

## About this code
This code was developed for Tensorflow 1.14 with Python 3.

## How to run

### Get the dataset
Amazon review dataaset can be downloaded from [here](http://jmcauley.ucsd.edu/data/amazon/index.html). As an example, you can download one dataset, `reviews_Automotive_5.json` to `data` folder from Automotive category with 5-core dataset in order to test running the code. 

### Run training
To train your model, run:

```
python main.py --data_path=/path/to/downloaded/gz/file --log_root=/path/to/a/log/directory
```
This will create a subdirectory of your specified `log_root` with timestamp in the format of `%Y%m%d-%H%M%S` where all checkpoints and other data will be saved. Then the model will start training.

Following parameters can be specified.
1. **vocab_size** (integer): The size of vocabulary (deafult 5000)
2. **emb_dim** (integer): The dimension of latent space (default 5)
3. **max_iter_steps** (integer): Maximum number of iterations for L-BFGS gradient descent step (default 50)
4. **num_iter_steps** (integer): Maximum number of iterations for minimizing the total loss (default 50)
5. **init_stddev** (float): Standard deviation of normal distribution to initialize variables (default 0.1)
6. **min_kappa, max_kappa** (float): Minimum and maximum value to initialize kappa variable (deafult 0.1 and 10.0)
7. **mu** (float): Parameter that trade-offs rating error and corpus likelihood (default 0.1)
8. **threshold** (float): Threshold for the convergence of Phi and Theta variables (default 1e-3)
9. **data_path** (string): Path to input data (default 'data/reviews_Automotive_5.json.gz')
10. **log_root** (string): Root directory for all logging (deafult 'logs/')

While the model is trained, all variables are trained with fixed topic of words, `z` variable, (Gradient Descent Step) and the topic is sampled with fixed variables (Gibbs Sampling). Gradient descent step uses L-BFGS method and is run for at most `max_iter_steps` iterations. After variables are trained from gradient descent step, validation and test errors are computed and printed.

### Tensorboard
Run Tensorboard from the logs directory. You should be able to see data from the training process.
```
tensorboard --logdir=logs\
```