import subprocess
import time
import csv

def run_and_time(command):
    start = time.perf_counter_ns()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter_ns()
    return (end - start) / 1e9  # Convert ns to seconds

def benchmark(executable, label, runs, lattice_length, writer):
    for i in range(1, runs + 1):
        duration = run_and_time(f"./escg -l {lattice_length}")
        writer.writerow([label, i, lattice_length, f"{duration:.9f}"])
        print(f"{label},{i},{lattice_length},{duration:.9f}")

with open("results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    # writer.writerow(["implementation", "run", "length", "time_seconds"])

    # benchmark("escg", "single-threaded", 10, 100, writer)
    # benchmark("escg", "single-threaded", 10, 200, writer)
    # benchmark("escg", "single-threaded", 10, 300, writer)
    
    # benchmark("escg", "single-threaded", 10, 400, writer)
    # benchmark("escg", "single-threaded", 10, 800, writer)
    benchmark("escg", "single-threaded", 1, 500, writer)
    benchmark("escg", "single-threaded", 1, 600, writer)
    benchmark("escg", "single-threaded", 1, 700, writer)
    