#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsedMilliseconds() const;
    double elapsedSeconds() const;
    double elapsedNanoseconds() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    bool running;
};

#endif // TIMER_HPP
