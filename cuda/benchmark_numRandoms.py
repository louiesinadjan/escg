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
        duration = run_and_time(f"./{executable} -l 400 -x true -r {numRandoms}")
        writer.writerow([numRandoms, i, f"{duration:.9f}"])
        print(f"{numRandoms},{i},{duration:.9f}")

with open("results_numrandoms.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["numRandoms", "run", "time_seconds"])
    
    benchmark("cuda-escg", 10, "1000000", writer)
    benchmark("cuda-escg", 10, "5000000", writer)
    benchmark("cuda-escg", 10, "10000000", writer)
    benchmark("cuda-escg", 10, "20000000", writer)
    benchmark("cuda-escg", 10, "30000000", writer)
    benchmark("cuda-escg", 10, "40000000", writer)
    benchmark("cuda-escg", 10, "50000000", writer)
    