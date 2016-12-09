#pragma once
#include <cstdint>
#include <queue>

#include <boost/thread.hpp>

namespace adrianyu
{
struct WordContextPair
{
    uint64_t word;
    uint64_t context;
};

struct DocAWord
{
    uint64_t doc_id;
    uint64_t word_id;
    int64_t word_ts;
};

struct AWord
{
    uint64_t word_id;
    int64_t word_ts;
};

class KahanSumation
{
    /*
    Kahan Summation Algorithm, see ref.:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    */
public:
    explicit KahanSumation(void) {
        clear();
    }
    explicit KahanSumation(const double val) {
        clear();
        sum = val;
    }

    void clear(void) {
        sum = 0;
        c = 0;
    }
    double get(void) const {
        return sum;
    }

    KahanSumation operator=(const double rhs_d) {
        clear();
        sum = rhs_d;
        return *this;
    }

    const KahanSumation& add(const double val) {
        const double y = val - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
        return *this;
    }
    KahanSumation & operator+=(const double rhs_d) {
        this->add(rhs_d);
        return *this;
    }
    friend KahanSumation operator+(KahanSumation lhs, const double rhs_d) {
        lhs += rhs_d;
        return lhs;
    }
    friend KahanSumation operator+(const double lhs_d, KahanSumation rhs) {
        rhs += lhs_d;
        return rhs;
    }

    KahanSumation & operator+=(const KahanSumation & rhs) {
        this->add(rhs.sum);
        this->add(rhs.c);
        return *this;
    }
    friend KahanSumation operator+(KahanSumation lhs, const KahanSumation &rhs) {
        lhs += rhs;
        return lhs;
    }

protected:
    double sum;
    double c;
};

template<class T>
class Queue {
public:
    Queue() {}
    Queue(const Queue &rhs) {
        this->_q = rhs._q;
    }
    void push(const T &item) {
        boost::lock_guard<boost::mutex> lk(_mut);
        _q.push(item);
        _cv.notify_one();
    }

    void pop(T &item) {
        boost::unique_lock<boost::mutex> lk(_mut);
        _cv.wait(lk, [this] {return !_q.empty(); });
        item = _q.front();
        _q.pop();
    }

    size_t size(void) {
        return _q.size();
    }
private:
    boost::mutex _mut;
    boost::condition_variable _cv;
    std::queue<T> _q;
};

}



