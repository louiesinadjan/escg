import subprocess
import time
import csv

def run_and_time(command):
    start = time.perf_counter()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter()
    return end - start

def benchmark(executable, runs, numRandoms, writer):
    for i in range(1, runs + 1):
        duration = run_and_time(f"./{executable} -l 400 -x true -R {numRandoms}")
        writer.writerow([numRandoms, i, f"{duration:.9f}"])
        print(f"{numRandoms},{i},{duration:.9f}")

with open("results_numrandoms.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    # writer.writerow(["numRandoms", "run", "time_seconds"])
    
    # benchmark("cuda-escg", 5, "1000000", writer)
    # benchmark("cuda-escg", 5, "5000000", writer)
    # benchmark("cuda-escg", 5, "10000000", writer)
    # benchmark("cuda-escg", 5, "20000000", writer)
    # benchmark("cuda-escg", 5, "30000000", writer)
    # benchmark("cuda-escg", 5, "40000000", writer)
    # benchmark("cuda-escg", 5, "50000000", writer)
    # benchmark("cuda-escg", 5, "50000000", writer)
    benchmark("cuda-escg", 5, "80000000", writer)
    benchmark("cuda-escg", 5, "90000000", writer)
    # benchmark("cuda-escg", 10, "100000000", writer)
    # benchmark("cuda-escg", 10, "250000000", writer)
    # benchmark("cuda-escg", 10, "500000000", writer)

    # benchmark("cuda-escg", 10, "27500000", writer)
    # benchmark("cuda-escg", 10, "32500000", writer)
    # benchmark("cuda-escg", 10, "60000000", writer)
    # benchmark("cuda-escg", 10, "70000000", writer)


    