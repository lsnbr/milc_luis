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




def stat_error(j_replicas : np.ndarray, avg : float) -> float:
    '''calculates the statistical errors from the jackknife replicas and the average value'''

    n = j_replicas.size
    return np.sqrt(
        ((j_replicas - avg)**2).sum() * (n-1) / n
    )




def search_uncorr(data : np.ndarray) -> None:
    '''plots statistical error over bin size, thus the plateau signals where the data becomes uncorrelated'''

    bin_sizes = list(range(1, data.size // 10 + 1))
    errors = []
    
    avg = data.mean()
    for b in bin_sizes:
        binned = bin_data(data, b)
        j_replicas = jackknife_replicas(binned)
        errors.append(stat_error(j_replicas, avg))

    plt.scatter(bin_sizes, errors)
    plt.xlabel('Bin Size')
    plt.ylabel('Error')
    plt.show()



