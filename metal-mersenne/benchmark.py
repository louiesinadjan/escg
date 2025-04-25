import subprocess
import time
import csv

def run_and_time(command):
    start = time.perf_counter_ns()
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.perf_counter_ns()
    return (end - start) / 1e9  # Convert ns to seconds

def benchmark(executable, label, runs, writer):
    for i in range(1, runs + 1):
        duration = run_and_time(f"./build/{executable}")
        writer.writerow([label, i, f"{duration:.9f}"])
        print(f"{label},{i},{duration:.9f}")

with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["implementation", "run", "time_seconds"])

    benchmark("metal-mersenne", "metal", 10, writer)
    benchmark("serial-mersenne", "serial", 10, writer)
