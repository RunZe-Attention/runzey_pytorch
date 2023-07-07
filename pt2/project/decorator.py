def double(x):
    return x * 2

def triple(x):
    return x * 3


def calc_numbet(func,x):
    return func(x)


print(calc_numbet(double,3))
print(calc_numbet(triple,3))

# 函数的返回值也可以是一个函数
def get_multiple_func(n):
    def multiple(x):
        return x * n

    return multiple

print(get_multiple_func(3)(2))
print(get_multiple_func(3)(3))



import time

def timeit(f):
    def wrapper(x):
        print(x)
        start_time = time.time()
        print(f)
        ret = f(x)
        end_time = time.time()
        print(end_time - start_time)
        return ret

    return wrapper

@timeit
def myfunc(x):
    time.sleep(x)

myfunc(10)

@timeit
def other_func(x):
    return x * 2

print(other_func(2))

def timeititer(iteration):
    def inner(f):
        def wrapper(*args,**kwargs):
            start_time = time.time()
            ret = f(*args,**kwargs)
            end_time = time.time()
            print(end_time-start_time)
            return ret
        return wrapper

@timeit(1000)
def double(x):
    return x * 2

double(2)




