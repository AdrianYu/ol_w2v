#pragma once
#include <vector>
#include <utility>
#include <deque>
#include <tuple>
#include <random>
#include <algorithm>
#include <valarray>
#include <numeric>
#include <cassert>
#include <chrono>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <memory>

#include <boost/thread.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/atomic.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/unordered_map.hpp>

#include <tbb/concurrent_queue.h>

#include "common_obj.h"
#include "common_utils.h"

namespace adrianyu
{

class WordSampler
{
public:
    explicit WordSampler() : _uni_01_dist(0, 1) {
        _alias_ptr = &_alias1;
        _alias_bk_ptr = &_alias2;
        _words_ptr = &_words1;
        _words_bk_ptr = &_words2;

        // the counter should dispatch by id
        _counter_num = 4;   // counter thread number
        _input_qsize = 15000;
        _counter_queues.resize(_counter_num);
        for (auto &q : _counter_queues) {
            q.set_capacity(_input_qsize);
        }
        for (size_t i = 0; i < _counter_num; ++i) {
            _counter_thrds.push_back(std::make_shared<boost::thread>(\
                WordSampler::WordCounter, this, i));
        }

        _symm = true;
        _window = 8;
        _subsampl = 1e-5;
        _subsamp_s = 1e-5;
        _subsamp_base = 1e9;
        // the doc-in should dispatch by id
        _dw_expire_ts = 3600 * 24 * 5;
        _doc_in_num = 4;   // doc in thread number
        _doc_in_queues.resize(_doc_in_num);
        for (auto &q : _doc_in_queues) {
            q.set_capacity(_input_qsize);
        }
        for (size_t i = 0; i < _doc_in_num; ++i) {
            _doc_in_thrds.push_back(std::make_shared<boost::thread>(\
                WordSampler::DocInput, this, i));
        }

        _alias_timer_interval = 1000;  // in milliseconds
        _alias_proc_num_thrshd = 20000; // proc number
        _min_wc_alias = 5;
        _p_factor = 0.75;
        _alias_timer_thrd = std::make_shared<boost::thread>(WordSampler::AliasGenTimer, this);

        _disp_num = 15;
        _pos_qsize = 100000;
        _w_ctx_queues.resize(_disp_num);
        for (auto &q : _w_ctx_queues) {
            q.set_capacity(_pos_qsize);
        }
       
        _acc_counts = 0;
        _total_counts = 0;
    }

    // input a word
    void enqueue(const DocAWord &dw) {
        size_t qidx = dw.doc_id % _doc_in_queues.size();
        _counter_queues[qidx].push(dw);
        _doc_in_queues[qidx].push(dw);
    }

    // it is better to use more queues to constrain the threads allowed to train specific word
    void get_word_context(WordContextPair &wc_pair, const size_t idx) {
        _w_ctx_queues[idx].pop(wc_pair);
    }

    // the iterator should be larger than / equal to num
    template <class InputIterator>
    InputIterator get_neg(const size_t num, InputIterator beg) {
        static thread_local std::mt19937_64 rnd_gen(\
            boost::hash<boost::thread::id>()(boost::this_thread::get_id()));
        static thread_local uint64_t proc_count = 0;
        static thread_local std::uniform_real_distribution<double> u01_dist;

        if (proc_count > 1000000) {
            //unsigned seed = boost::hash<boost::thread::id>()(boost::this_thread::get_id());
            unsigned seed = static_cast<unsigned>(100 * u01_dist(rnd_gen));
            rnd_gen.seed(boost::posix_time::microsec_clock::local_time(\
                ).time_of_day().total_milliseconds() + seed);
            proc_count = 0;
        }

        std::vector<std::tuple<size_t, size_t, double> > *alias = NULL;
        std::vector<uint64_t> *words = NULL;
        {
            boost::shared_lock<boost::shared_mutex> lock(_buffer_mutex); // double buffer, should not need this.
            alias = _alias_ptr;
            words = _words_ptr;
        }
        for (size_t i = 0; i < num; ++i) {
            double bin_p = u01_dist(rnd_gen);
            u01_dist(rnd_gen);
            double uni_p = u01_dist(rnd_gen);
            const size_t widx = WordSampler::SampleAlias(*alias, bin_p, uni_p);
            if (widx == -1) {
                break;
            }
            *beg = (*words)[widx];
            beg++;
            u01_dist(rnd_gen);
        }
        proc_count++;
        return beg;
    }

    /*
     *  since this function will acquire write lock on counter and doc storage,
     *  it is recommended not to call this function as often. A daily operation should suffice.
    */
    void del_expire(const int64_t expire_ts, std::vector<uint64_t> &words_del, std::vector<uint64_t> &docs_del) {
        // delete expired words
        for (auto &word : _word_ts) {
            if (word.second < expire_ts) {
                words_del.push_back(word.first);
            }
        }
        uint64_t counts = 0;
        boost::unique_lock<boost::shared_mutex> counter_lock(_counter_mutex);
        for (auto word : words_del) {
            _word_ts.unsafe_erase(word);
            counts += _word_counts[word];
            _word_counts.unsafe_erase(word);
        }
        counter_lock.unlock();
        _total_counts -= counts;    // update the _total_counts

        // delete expired doc
        for (auto &doc : _doc_ts) {
            if (doc.second < expire_ts) {
                docs_del.push_back(doc.first);
            }
        }
        boost::unique_lock<boost::shared_mutex> docin_lock(_doc_in_mut);
        for (auto &doc : docs_del) {
            _doc_ts.unsafe_erase(doc);
            _doc_words.unsafe_erase(doc);
        }
        
    }

protected:
    size_t _counter_num;
    boost::atomic<int64_t> _acc_counts;
    boost::atomic<uint64_t> _total_counts;
    std::vector<tbb::concurrent_bounded_queue<DocAWord> > _counter_queues;
    boost::shared_mutex _counter_mutex;
    tbb::concurrent_unordered_map<uint64_t, int64_t> _word_ts;
    tbb::concurrent_unordered_map<uint64_t, uint64_t> _word_counts;
    std::vector<std::shared_ptr<boost::thread> > _counter_thrds;

    size_t _disp_num;
    bool _symm;
    int _window;
    double _subsampl;
    double _subsamp_base;
    double _subsamp_s;
    size_t _doc_in_num;
    boost::shared_mutex _doc_in_mut;
    size_t _input_qsize;
    size_t _pos_qsize;
    int64_t _dw_expire_ts;  // used to expire too old clicks
    tbb::concurrent_unordered_map<uint64_t, int64_t> _doc_ts;
    tbb::concurrent_unordered_map<uint64_t, boost::circular_buffer<AWord> > _doc_words;
    //std::unordered_map<uint64_t, int64_t> _user_ts;
    std::vector<tbb::concurrent_bounded_queue<DocAWord> > _doc_in_queues;
    std::vector<std::shared_ptr<boost::thread> > _doc_in_thrds;
    std::vector<tbb::concurrent_bounded_queue<WordContextPair> > _w_ctx_queues;

    int64_t _alias_timer_interval;
    int64_t _alias_proc_num_thrshd;
    uint64_t _min_wc_alias;
    std::shared_ptr<boost::thread> _alias_timer_thrd;
    //boost::shared_mutex _alias_mutex;

    std::mt19937_64 _rnd_gen;
    std::uniform_real_distribution<double> _uni_01_dist;
    double _p_factor;
    // double buffer
    boost::shared_mutex _buffer_mutex;
    std::vector<std::tuple<size_t, size_t, double> > * _alias_ptr;
    std::vector<std::tuple<size_t, size_t, double> > * _alias_bk_ptr;
    std::vector<std::tuple<size_t, size_t, double> > _alias1;
    std::vector<std::tuple<size_t, size_t, double> > _alias2;
    std::vector<uint64_t> * _words_ptr;
    std::vector<uint64_t> * _words_bk_ptr;
    std::vector<uint64_t> _words1;
    std::vector<uint64_t> _words2;

    static void GenAlias(const std::vector<double> & probs,
        std::vector<std::tuple<size_t, size_t, double> > & Alias)
    {
        const double uni_prob = 1.0 / static_cast<double>(probs.size());
        KahanSumation prob_p_sum_kh;
        for (auto p : probs) {
            prob_p_sum_kh += p;
        }
        const double prob_p_sum = prob_p_sum_kh.get();
        std::deque<std::pair<size_t, double> > L;
        std::deque<std::pair<size_t, double> > H;
        double rprob = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            rprob = probs[i] / prob_p_sum;
            if (rprob <= uni_prob) {
                L.push_back(std::make_pair(i, rprob));
            }
            else {
                H.push_back(std::make_pair(i, rprob));
            }
        }
        Alias.clear();
        Alias.reserve(probs.size());
        while (!L.empty() && !H.empty()) {
            const std::pair<size_t, double> & L_item = L.front();
            const std::pair<size_t, double> & H_item = H.front();
            Alias.push_back(std::make_tuple(L_item.first, H_item.first, L_item.second));
            const double prob_rsd = H_item.second + L_item.second - uni_prob;
            if (prob_rsd > uni_prob) {
                H.push_back(std::make_pair(H_item.first, prob_rsd));
            }
            else {
                L.push_back(std::make_pair(H_item.first, prob_rsd));
            }
            H.pop_front();
            L.pop_front();
        }
        // if any of the H/L is not empty, we fill Alias with the same index.
        while (!L.empty()) {
            Alias.push_back(std::make_tuple(L.front().first, L.front().first, L.front().second));
            L.pop_front();
        }
        while (!H.empty()) {
            Alias.push_back(std::make_tuple(H.front().first, H.front().first, H.front().second));
            H.pop_front();
        }
    }

    size_t SampleAlias(const std::vector<std::tuple<size_t, size_t, double> > & Alias) {
        _uni_01_dist.reset();
        const double bin_p = _uni_01_dist(_rnd_gen);
        size_t bin = static_cast<size_t>(bin_p * static_cast<double>(Alias.size()));
        auto & tup = Alias[bin];
        _uni_01_dist.reset();
        const double uni_p = _uni_01_dist(_rnd_gen);
        const double lp = static_cast<double>(Alias.size()) * std::get<2>(tup);
        // in real application, it is common that almost all prob in probs is near zero
        // hence, the value of lp is small.
        if (lp < uni_p) {
            return std::get<1>(tup);
        }
        else {
            return std::get<0>(tup);
        }
    }

    // bin_p & uni_p is two random number
    static size_t SampleAlias(const std::vector<std::tuple<size_t, size_t, double> > &Alias,\
        const double bin_p, const double uni_p) {
        if (Alias.empty()) {
            return -1;
        }
        size_t bin = static_cast<size_t>(bin_p * static_cast<double>(Alias.size()));
        auto & tup = Alias[bin];
        const double lp = static_cast<double>(Alias.size()) * std::get<2>(tup);
        // in real application, it is common that almost all prob in probs is near zero
        // hence, the value of lp is small.
        if (lp < uni_p) {
            return std::get<1>(tup);
        }
        else {
            return std::get<0>(tup);
        }
    }

    static void WordCounter(WordSampler *ws, size_t idx) {
        std::cerr << "thread index [" << idx << "] begins to run." << std::endl;
        DocAWord word;
        while (true) {
            ws->_counter_queues[idx].pop(word);
            //boost::upgrade_lock<boost::shared_mutex> lock(ws->_counter_mutex);
            boost::shared_lock<boost::shared_mutex> lock(ws->_counter_mutex);
            if (!ws->_word_counts.count(word.word_id)) {
                // get write lock
                //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
                ws->_word_counts[word.word_id] = 0;
                ws->_word_ts[word.word_id] = word.word_ts;
            }
            if (ws->_word_ts[word.word_id] < word.word_ts) {
                ws->_word_ts[word.word_id] = word.word_ts;
            }
            ws->_word_counts[word.word_id]++;
            ws->_acc_counts++;
            ws->_total_counts++;
            double sub_p = ws->_subsamp_base / ws->_total_counts;
            if (sub_p < 1) {
                ws->_subsampl = ws->_subsamp_s;
            }
            else {
                ws->_subsampl = sub_p * ws->_subsamp_s;
            }

            if (0 == ws->_total_counts % 1000000) {
                std::cerr << "at timestamp: " << cur_ts()
                    << ", words processed: " << ws->_total_counts << std::endl;
            }
        }
    }

    static void AliasGenTimer(WordSampler *ws) {
        int64_t pre_time = \
            boost::posix_time::microsec_clock::local_time().time_of_day().total_milliseconds();
        while (true) {
            while (boost::posix_time::microsec_clock::\
                local_time().time_of_day().total_milliseconds()\
                - pre_time < ws->_alias_timer_interval \
                && ws->_acc_counts < ws->_alias_proc_num_thrshd) {
                boost::this_thread::sleep(boost::posix_time::milliseconds(10));
            }
            tbb::concurrent_unordered_map<uint64_t, uint64_t> word_counts;
            {
                // read lock, copy current word counts
                boost::shared_lock<boost::shared_mutex> lock(ws->_counter_mutex);
                word_counts = ws->_word_counts;
            }
            ws->_acc_counts = 0;
            pre_time = \
                boost::posix_time::microsec_clock::local_time().time_of_day().total_milliseconds();
            ws->_words_bk_ptr->resize(word_counts.size());
            std::vector<double> probs(word_counts.size());
            size_t widx = 0;
            uint64_t total_w_counts = 0;
            for (const auto &item : word_counts) {
                if (item.second < ws->_min_wc_alias) {  // if the count of current word is small, we don't add to alias
                    continue;
                }
                (*ws->_words_bk_ptr)[widx] = item.first;
                probs[widx] = std::pow(item.second, ws->_p_factor);
                total_w_counts += item.second;
                widx++;
            }
            // we need to shrink the pre-allocated vector
            ws->_words_bk_ptr->resize(widx);
            probs.resize(widx);
            // generate alias
            uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
            ws->_rnd_gen.seed(seed);

            GenAlias(probs, *ws->_alias_bk_ptr);

            // switch current and backup ptr
            {
                boost::unique_lock<boost::shared_mutex> lock(ws->_buffer_mutex); // the following procedure should be quite atomic
                auto wordsptr = ws->_words_ptr;
                ws->_words_ptr = ws->_words_bk_ptr;
                ws->_words_bk_ptr = wordsptr;
                auto aliasptr = ws->_alias_ptr;
                ws->_alias_ptr = ws->_alias_bk_ptr;
                ws->_alias_bk_ptr = aliasptr;
            }
        }
    }

    static void DocInput(WordSampler *ws, const size_t idx) {
        std::cerr << "click processor DocInput thread index [" << idx << "] begins to run." << std::endl;

        // this function is quite different from a normal thread pool, so thread_local is not needed.
        std::mt19937_64 rnd_gen(\
            boost::hash<boost::thread::id>()(boost::this_thread::get_id()));
        uint64_t proc_count = 0;
        std::uniform_real_distribution<double> u01_dist;

        DocAWord dw;
        WordContextPair wcp1;
        WordContextPair wcp2;
        int64_t cur_ts;
        while (true) {

            if (proc_count > 1000000) {
                unsigned seed = static_cast<unsigned>(100 * u01_dist(rnd_gen));
                rnd_gen.seed(boost::posix_time::microsec_clock::local_time(\
                    ).time_of_day().total_milliseconds() + seed);
                proc_count = 0;
            }

            ws->_doc_in_queues[idx].pop(dw);
            cur_ts = dw.word_ts;

            // The sub-sampling randomly discards frequent words while keeping the ranking same
            if (ws->_subsampl > 0) {
                double word_c = 0;
                boost::shared_lock<boost::shared_mutex> lock(ws->_counter_mutex);
                if (ws->_word_counts.count(dw.word_id)) {
                    word_c = static_cast<double>(ws->_word_counts[dw.word_id]);
                }
                lock.unlock();
                const double cd_tmp = word_c / (ws->_subsampl * static_cast<double>(ws->_total_counts));
                double prob = (std::sqrt(cd_tmp) + 1.0) / cd_tmp;
                if (!std::isfinite(prob)) {
                    prob = 1.0;
                }
                //std::cerr << prob << "\t" << word_c << "\t"
                //    << ws->_subsampl << "\t" << ws->_total_counts << std::endl;
                // discard
                if (prob < u01_dist(rnd_gen)) {
                    continue;
                }
            }
            proc_count++;

            //boost::upgrade_lock<boost::shared_mutex> lock(ws->_doc_in_mut);
            boost::shared_lock<boost::shared_mutex> lock(ws->_doc_in_mut);
            if (!ws->_doc_words.count(dw.doc_id)) {
                // get write lock
                //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
                ws->_doc_words.emplace(dw.doc_id, boost::circular_buffer<AWord>(ws->_window));
                ws->_doc_ts[dw.doc_id] = dw.word_ts;
            }
            if (ws->_doc_ts[dw.doc_id] < dw.word_ts) {
                ws->_doc_ts[dw.doc_id] = dw.word_ts;
            }
            if (!ws->_doc_words[dw.doc_id].empty()) {
                wcp1.word = dw.word_id;
                wcp2.context = dw.word_id;
                size_t wcp1_disp_idx = wcp1.word % ws->_disp_num;
                for (size_t i = 0; i < ws->_doc_words[dw.doc_id].size(); ++i) {
                    wcp1.context = ws->_doc_words[dw.doc_id][i].word_id;
                    wcp2.word = ws->_doc_words[dw.doc_id][i].word_id;
                    // only generate pairs if stored data is not invalid
                    if (ws->_doc_words[dw.doc_id][i].word_ts > cur_ts - ws->_dw_expire_ts) {
                        // the positive training sample queue
                        ws->_w_ctx_queues[wcp1_disp_idx].push(wcp1);
                        if (ws->_symm) {
                            ws->_w_ctx_queues[wcp2.word % ws->_disp_num].push(wcp2);
                        }
                    }
                    else {
                        // do nothing here. note that I don't delete anything
                        ;
                    }
                }
            }
            ws->_doc_words[dw.doc_id].push_back({ dw.word_id, dw.word_ts });
        }
    }

};

}
