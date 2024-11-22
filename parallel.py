import multiprocessing as mp
from multiprocessing import Pool #Pool object represent pool, it schedules the process, n of
import time
import math
#import numpy as np


#Try to not use threads at the same time
#Threads: Can talk to each other, run in share memory space
#Processes: Isolated object, different memory spaces

#Use object as arguments as long as they are serializable
#Parallel with bashm python plot_1.py & python plot_2.py, & -> Parallel, && -> serial


A = []
B = []
C = []

#Think about independent things that can be done at the same time
def calculation_one(num):
    for n in num:
        A.append(math.sqrt(n**3))

def calculation_two(num):
    for n in num:
        B.append(math.sqrt(n**4))

def calculation_three(num):
    for n in num:
        C.append(math.sqrt(n**5))

# Helper function to execute a function with its arguments
def execute_function(func, args):
    return func(args)


if __name__ == '__main__':

    #Set number for operations
    number_list = list(range(5000000))

    # Set parallel with starmap:

    #Start time recording
    start = time.time()

    #with: use contezt manager, is something that closes and open, no need to close

    # Create a Pool with 3 processes (one for each task), if processes = None: max out # processes
    with mp.Pool(processes=None) as pool:
        # Prepare a list of functions and arguments for starmap
        tasks = [
            (calculation_one, number_list),
            (calculation_two, number_list),
            (calculation_three, number_list)
        ]

        # Use starmap to apply each function with its arguments
        results = pool.starmap(execute_function, tasks)

    end = time.time()
    print("Starmap Parallel: ", end - start)
    

    # Similar to threards do processes, set the target function and the arguments of it as tuple
    p1 = mp.Process(target = calculation_one, args=(number_list, ))
    p2 = mp.Process(target = calculation_two, args=(number_list, ))
    p3 = mp.Process(target = calculation_three, args=(number_list, ))

    #Start time recording
    start = time.time()
    p1.start()
    p2.start()
    p3.start()
    end = time.time()
    print("Parallel: ", end - start)

    #Set results
    parallel_A = A
    parallel_B = B
    parallel_C = C



    #Start time recording
    start = time.time()

    #Do calculations
    calculation_one(number_list)
    calculation_two(number_list)
    calculation_three(number_list)

    #Finish time
    end = time.time()

    print("Non parallel: ", end-start)


    #Check if same:
    print(A == parallel_A)
    print(B == parallel_B)
    print(C == parallel_C)
