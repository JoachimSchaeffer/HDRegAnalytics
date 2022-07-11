# Module to generate synthetic data
# Author: Joachim Schaeffer, joachim.schaeffer@posteo.de

import numpy as np
import random
# Initialize the random seed to ensure reproducibility of the results in the paper
random.seed(42)

def generate_wide_datamatrix(fun, range_x, range_m, columns, rows, snr_x=-1):
    """Generates synthethic data to test the linearization methodology. 
    Arguments:
    --------
    fun: function that is defined for all values inside range. Representing a measurement that was taken from any process. 
    range_x: np.array with two elements, the first being strictly smaller than the second, defining the meaning of the columsn
    range_m: range of factor
        each row of x is equal to m*fun(x), where m is sampled from a uniform distribution
    datapoints: number of linearly spaced datapoints for evaluating fun, equal to number of columns of X
    rows: number of rows of X, i.e. 
    snr_x: Signal to noise ratio of AWGN to be added on the signal, if -1, then this functions adds no noise to the signal
    Returns: X
    """

    # Sample m
    m = np.random.uniform(low=range_m[0], high=range_m[1], size=rows)

    # create X and y
    x = np.linspace(range_x[0], range_x[1], columns)
    fun_values = fun(x)
    X = np.zeros([rows, columns])
    y = np.zeros(rows)
    for i in range(rows):
        row_i = m[i]*fun_values
        if snr_x != -1: 
            # Add Gaussian noise to the measurements
            # Snippet below partly copied/adapted/inspired by: 
            # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
            # Answer from Noel Evans, accessed: 18.05.2022, 15:37 CET
            # Calculate signal power and convert to dB 
            sig_avg_watts = np.mean(row_i**2)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - snr_x
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), columns)
            # Noise up the original signal
            row_i += noise

        X[i, :] = row_i

    return np.array(X), x

def generate_target_values(X, targetfun, percentage_range_x_to_t=[0,1]):
    '''This function takes the wide data matix X as an input and generates target function values y based on the defined functions
    X: \in R^mxn with n>>m (wide data matrix, used as input for the targetfunction y
    targetfun: Underlying relationship between X and y. This can be any function from R^n -> R^1
        This is also the ideal feature for predicting y and thus the information we would like to discover by applying the lionearization methodology. 
    percentage_range_x_to_t: array with two elements, the first one being strictly smaller than the second value, both being strcitly between 0 and 1, 
        defines the range of input data that shall be used to generate the target function
        The reson behind this is that in process data analytics often a sitation can arise where only a part of the data is relevant to predict the target y  
    '''
    rows = X.shape[0]
    columns = X.shape[1]
    y = np.zeros([rows])
    for i in range(rows):
        row_i = X[i, :]
        low_ind = int(percentage_range_x_to_t[0]*columns)
        high_ind = int(percentage_range_x_to_t[1]*columns)
        y[i] = targetfun(row_i[low_ind:high_ind])
    
    return y


