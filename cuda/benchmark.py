import subprocess
import time
import csv

def run_and_time(command):
    start = time.perf_counter()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter()
    return end - start

def benchmark(executable, label, runs, lattice_length, maxstep, writer):
    for i in range(1, runs + 1):
        duration = run_and_time(f"./{executable} -l {lattice_length} -h {lattice_length} -x {maxstep} -R 40000000")
        writer.writerow([label, i, lattice_length, f"{duration:.9f}"])
        print(f"{label},{i},{lattice_length},{duration:.9f}")

with open("results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    # writer.writerow(["implementation", "run", "length", "time_seconds"])
    
    # benchmark("cuda-escg", "cuda", 10, 100, "false", writer)
    # benchmark("cuda-escg", "cuda-max", 10, 100, "true", writer)
    
    # benchmark("cuda-escg", "cuda", 10, 200, "false", writer)
    # benchmark("cuda-escg", "cuda-max", 10, 200, "true", writer)
    
    # benchmark("cuda-escg", "cuda", 10, 300, "false", writer)
    # benchmark("cuda-escg", "cuda-max", 10, 300, "true", writer)
    
    # benchmark("cuda-escg", "cuda", 10, 400, "false", writer)
    # benchmark("cuda-escg", "cuda-max", 10, 400, "true", writer)

    benchmark("cuda-escg", "cuda", 1, 500, "false", writer)
    benchmark("cuda-escg", "cuda-max", 1, 500, "true", writer)
    
    # benchmark("cuda-escg", "cuda", 10, 600, "false", writer)
    # benchmark("cuda-escg", "cuda-max", 10, 600, "true", writer)

    benchmark("cuda-escg", "cuda", 1, 700, "false", writer)
    benchmark("cuda-escg", "cuda-max", 1, 700, "true", writer)
    
    # benchmark("cuda-escg", "cuda", 10, 800, "false", writer)

    # benchmark("cuda-escg", "cuda-max", 5, 1600, "true", writer)

    # benchmark("cuda-escg", "cuda-max", 1, 3200, "true", writer)
