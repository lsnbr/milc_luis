import numpy as np
import matplotlib.pyplot as plt







def bin_data(data : np.ndarray, b : int) -> np.ndarray:
    '''group into bins of size b and average those'''

    return np.array([ data[i*b : (i+1)*b].sum() / b
                      for i in range(0, data.size // b) ])




def jackknife_replicas(data : np.ndarray) -> np.ndarray:
    '''for N data points, create the N jackknife replicas'''

    data_sum = data.sum()
    return np.array([ (data_sum - x) / (data.size - 1)
                      for x in data                    ])




def stat_error(data : np.ndarray, bin_size : int) -> float:
    '''calculates the statistical errors with jackknife replica method for given bin size'''

    binned_data = bin_data(data, bin_size)
    n = binned_data.size
    return np.sqrt(
        ((jackknife_replicas(binned_data) - data.mean())**2).sum() * (n-1) / n
    )




def search_uncorr(data : np.ndarray) -> None:
    '''plots statistical error over bin size, thus the plateau signals where the data becomes uncorrelated'''

    bin_sizes = list(range(1, data.size // 10 + 1))
    errors = [stat_error(data, b) for b in bin_sizes]

    plt.plot(bin_sizes, errors, marker='o')
    plt.xlabel('Bin Size')
    plt.ylabel('Error')
    plt.show()



