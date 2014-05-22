from problem_props import constants
import python_utils.python_utils.utils as utils
from python_utils.python_utils import caching
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import numpy as np

class datum(object):

    def __init__(self, propid, types_and_times):
        self.propid = propid
        self.types_and_times = types_and_times


@caching.default_cache_fxn_decorator
def get_katrine_processed_data():
    return pd.read_csv(constants.katrine_911_file, index_col = 0).sort('ENTRY_DT')


def get_katrine_propid_list():
    """
    dataframe of sizes, index is propid.  has width of 1
    """
    return pd.read_csv(constants.katrine_propid_list, index_col = 0, squeeze = True)


class katrine_propid_iterable(utils.DataIdIterable):

    def __init__(self, start_idx, end_idx):
        propids = get_katrine_propid_list()
        self.propids = propids.index[start_idx:end_idx+1]

    def __iter__(self):
        return iter(self.propids)


class propid_iterable_by_size(utils.DataIdIterable):

    def __init__(self, min_size, max_size):
        propids = get_katrine_propid_list()
        ok_ids = propids[[(x >= min_size and x < max_size) for x in propids]].index
        self.propids = ok_ids


    def __iter__(self):
        return iter(self.propids)


class ys_f(object):

    def __init__(self):
        self.d = get_katrine_processed_data()

    def __call__(self, propid):
        ans = self.d[self.d.propID == propid][['ENTRY_DT','type']].sort('ENTRY_DT')
        ans.columns = ['time', 'type']
        return ans


class get_data(object):

    def __init__(self, ys_f):
        self.ys_f = ys_f

    def __call__(self, propid_iterable):
        return [datum(propid, self.ys_f(propid)) for propid in propid_iterable]
        

def data_to_basic_model_pystan_input(data):
    """
    returns dictionary of data ys lengths, times, types
    adds 1 to type since indices in stan start at 1, not 0
    """
    lengths = [len(datum.types_and_times) for datum in data]
    times = pd.concat([datum.types_and_times.time for datum in data])
    types = pd.concat([datum.types_and_times.type + 1 for datum in data])
    return {'lengths':lengths, 'times':times, 'types':types}


@caching.default_cache_fxn_decorator
@caching.default_read_fxn_decorator
@caching.default_write_fxn_decorator
def get_basic_model_fit(prior_params, num_iters, num_chains, init_f, seed, the_type, data):
    """
    convention is to return by permuted and unpermuted traces
    """
    import problem_props.constants as constants
    import pystan
    stan_file = constants.basic_model_stan_file
    stan_data = data_to_basic_model_pystan_input(data)
    other_inputs = {\
        'n_types': max(stan_data['types']),\
            'n_props': len(data),\
            'n_events': len(stan_data['types']),\
            'the_type': the_type + 1\
            }
    all_stan_data = dict(prior_params.items() + stan_data.items() + other_inputs.items())
    if init_f == None:
        fit = pystan.stan(file = stan_file, data = all_stan_data, seed = seed, iter = num_iters, chains = num_chains, verbose = True)
    else:
        init_d = [init_f(data, i) for i in range(num_chains)]
        fit = pystan.stan(file = stan_file, data = all_stan_data, seed = seed, iter = num_iters, chains = num_chains, verbose = True, init = init_d)
    return fit.extract(permuted=True), fit.extract(permuted=False)


type_0_color = 'black'
type_1_color = 'red'
type_2_color = 'orange'
type_3_color = 'green'
type_4_color = 'blue'

type_to_color = {\
    0: type_0_color,\
        1: type_1_color,\
        2: type_2_color,\
        3: type_3_color,\
        4: type_4_color\
        }


def plot_timelines(propid_iterable, ys_f, min_time, max_time, num_per_fig):
    all_ys = [ys_f(propid) for propid in propid_iterable]
    num_chunks = (len(all_ys) / num_per_fig) + 1
    ys_chunks = utils.split_list_like(all_ys, num_chunks)
    import pdb
    figs = []
    size = 5
    for chunk in ys_chunks:
        fig, ax = plt.subplots()
        for i, ys in enumerate(chunk):
            colors = [type_to_color[t] for t in ys.type]
            ax.scatter(ys.time, [i for z in range(len(ys))], color = colors, s = size)
        ax.set_ylim((-1, len(chunk)))
        ax.set_xlabel('day index')
        ax.set_ylabel('property index')
        figs.append(fig)
    return figs

def plot_posterior_boxplots(traces):
    figs = []
    for key, val in traces.iteritems():
        if len(val.shape) == 1:
            fig, ax = plt.subplots()
            ax.hist(val)
            ax.set_title(key)
        elif len(val.shape) == 2:
            cols = [x for x in val.T]
            fig, ax = plt.subplots()
            ax.boxplot(cols)
            dim = len(cols)
            ax.set_xticks(range(0, dim))
            ax.set_xlabel('property type')
            ax.set_xticklabels(map(str, range(1, dim+1)), rotation = 'vertical')
            ax.set_title(key)
        else:
            assert False
        figs.append(fig)
    return figs
