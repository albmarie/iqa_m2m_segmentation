from class_utils import Correlation_mode
from scipy.stats.mstats import pearsonr, spearmanr, kendalltau

####################################################################################################
##########          FUNCTIONS          #############################################################
####################################################################################################

def get_correlation_fn(correlation: Correlation_mode):
    if correlation == Correlation_mode.PLCC:
        return lambda x, y:   pearsonr(x, y)
    elif correlation == Correlation_mode.SROC:
        return lambda x, y:  spearmanr(x, y)
    elif correlation == Correlation_mode.KROCC:
        return lambda x, y: kendalltau(x, y)
    else:
        raise Exception(f'Unsupported correlation mode ({correlation}).')

####################################################################################################

"""
Compute given metric between torch.Tensor tensor_x and tensor_y.
"""
def compute_correlation(x, y, correlation: Correlation_mode):
    assert len(x) == len(y)
    if any([score != score for score in x]) or any([score != score for score in y]): #If there is at least 1 NaN value...
        return float('Nan')

    correlation_fn = get_correlation_fn(correlation)
    correlation_rho, correlation_p = correlation_fn(x, y)
    return correlation_rho, correlation_p

####################################################################################################
##########          CORRELATION          ###########################################################
####################################################################################################

class Correlation_fn(object):
    def __init__(self, correlation: Correlation_mode):
        self.correlation = correlation

    def __call__(self, x, y):
        return compute_correlation(x, y, self.correlation)

####################################################################################################
####################################################################################################
####################################################################################################