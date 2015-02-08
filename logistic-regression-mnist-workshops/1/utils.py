import time

def timed(func):
    """ Decorator for easy time measurement """
    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        tend = time.time()
        print "{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), tend - tstart)
        return result

    return timed



cache_dict = {}
def cached(func):
    global cache_dict


    def func_caching(*args, **dict_args):
        key = (func.__name__, args, frozenset(dict_args.items()))
        if key in cache_dict:
            return cache_dict[key]
        else:
            returned_value = func(*args, **dict_args)
            cache_dict[key] = returned_value
            return returned_value

    return func_caching