#include "timer.hpp"

Timer::Timer() : running(false) {}

void Timer::start() {
    startTime = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::stop() {
    endTime = std::chrono::high_resolution_clock::now();
    running = false;
}

double Timer::elapsedNanoseconds() const {
    std::chrono::time_point<std::chrono::high_resolution_clock> endTimePoint;
    if (running) {
        endTimePoint = std::chrono::high_resolution_clock::now();
    } else {
        endTimePoint = endTime;
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(endTimePoint - startTime).count();
}

double Timer::elapsedMilliseconds() const { return elapsedNanoseconds() / 1000000.0; }

double Timer::elapsedSeconds() const { return std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime).count(); }