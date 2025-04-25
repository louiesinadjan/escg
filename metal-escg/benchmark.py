import subprocess
import time
import csv

def run_and_time(command):
    start = time.perf_counter_ns()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter_ns()
    return (end - start) / 1e9  # Convert ns to seconds

def benchmark(executable, label, runs, lattice_length, maxstep, writer):
    for i in range(1, runs + 1):
        duration = run_and_time(f"./build/{executable} -l {lattice_length} -h {lattice_length} -x {maxstep}")
        writer.writerow([label, i, lattice_length, f"{duration:.9f}"])
        print(f"{label},{i},{lattice_length},{duration:.9f}")

with open("results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    # writer.writerow(["implementation", "run", "length", "time_seconds"])

    # benchmark("metal", "metal-escg", 20, 200, writer)
    # benchmark("metal", "metal-escg", 20, 100, writer)
    
    # benchmark("metal", "metal-max", 10, 100, "true", writer)
    # benchmark("metal", "metal-max", 10, 200, "true", writer)
    # benchmark("metal", "metal-max", 10, 300, "true", writer)
    # benchmark("metal", "metal-max", 10, 400, "true", writer)
    
    benchmark("metal", "metal", 2, 500, "true", writer)
    benchmark("metal", "metal-max", 2, 500, "true", writer)
    
    # benchmark("metal", "metal", 5, 600, "true", writer)
    # benchmark("metal", "metal-max", 5, 600, "true", writer)
    