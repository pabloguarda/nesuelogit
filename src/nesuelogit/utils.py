from functools import wraps
import isuelogit
import time

import pesuelogit.networks


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def read_paths(block_output = True, **kwargs):

    if block_output:
        with isuelogit.printer.block_output(show_stdout=False, show_stderr=False):
            pesuelogit.networks.read_paths(**kwargs)
        print('Paths were read and incidence matrix were built')

    else:
        pesuelogit.networks.read_paths(**kwargs)

def load_k_shortest_paths(block_output = True, **kwargs):

    if block_output:
        with isuelogit.printer.block_output(show_stdout=False, show_stderr=False):
            pesuelogit.networks.load_k_shortest_paths(**kwargs)
        print('Paths were loaded and incidence matrix were built')

    else:
        pesuelogit.networks.load_k_shortest_paths(**kwargs)

