#pragma once

#include <random>
#include <chrono>
#include <cmath>
#include <ctime>

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sched.h>

namespace adrianyu
{

inline int get_tid(void) {
    return syscall(SYS_gettid);
}

inline int cur_ts(void) {
    return std::time(NULL);
}

inline double sigmoid(const double x) {
    return std::tanh(x / 2.0) / 2.0 + 0.5;
}

class SigTable {
public:

    // boundary should be positive
    void init(const double boundary, const size_t table_size) {
        _boundary = boundary;
        _data.resize(table_size);
        for (size_t i = 0; i < table_size; ++i) {
            double x = -boundary + 2 * i * boundary / table_size;
            _data[i] = sigmoid(x);
        }
    }
    double operator()(const double x) {
        const int idx = (x / _boundary + 1.0) / 2 * _data.size();
        if (idx < 0) {
            return 0;
        }
        else {
            if (static_cast<size_t>(idx) >= _data.size()) {
                return 1;
            }
            else {
                return _data[idx];
            }
        }
    }

private:
    double _boundary;
    std::vector<double> _data;
};

}
