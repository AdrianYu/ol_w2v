#pragma once

#include <utility>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>

#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>

#include <Eigen/Eigen>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_hash_map.h>

#include "common_utils.h"
#include "common_obj.h"
#include "WordSampler.h"

namespace adrianyu
{

class Word2Vec
{
    /*
        train word2vec using RMSprop algorithm
        // TODO: using SGD to train context embedding
    */
public:

    Word2Vec() :uni_dist(-1, 1)
    {
        _neg_num = 8;
        _embedding_dim = 128;
        _learning_rate = 0.001;
        _gamma = 0.9;

        sigtable.init(8.0, 10000);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        rnd_gen.seed(seed);

        _trainer_num = 15;
        for (size_t i = 0; i < _trainer_num; ++i) {
            _trainer_thrds.push_back(std::make_shared<boost::thread>(\
                Word2Vec::train_thrd, this, i));
        }

        _dumper_intv = 3600; // in seconds
        _dumper_timer_thrd = std::make_shared<boost::thread>(Word2Vec::dumper_timer, this);
        _expire_ts = 3600 * 24 * 6;
        _expire_intv = 3600 * 24;
        _expire_timer_thrd = std::make_shared<boost::thread>(Word2Vec::expire_timer, this);

        _sgd_lr = 0.02;
        _sgd_lr_min = _sgd_lr * 1e-4;

    }

    void input(std::istream &input_stream) {
        DocAWord dw;
        uint64_t seq_ts;
        uint64_t doc_id;
        uint64_t word_id;
        while (input_stream) {
            input_stream >> seq_ts >> doc_id >> word_id;
            dw.doc_id = doc_id;
            dw.word_id = word_id;
            dw.word_ts = seq_ts;
            _cur_ts = seq_ts;

            _word_sampler.enqueue(dw);
        }
    }

    void join(void) {
        for (size_t i = 0; i < _trainer_thrds.size(); ++i) {
            _trainer_thrds[i]->join();
        }
        _dumper_timer_thrd->join();
        _expire_timer_thrd->join();
    }

private:
    size_t _train_thrd_num;
    size_t _neg_num;
    int _embedding_dim;
    float _learning_rate;
    float _gamma;
    const float _epsilon = 1e-20;

    SigTable sigtable;

    WordSampler _word_sampler;

    std::mt19937_64 rnd_gen;
    std::uniform_real_distribution<double> uni_dist;

    float _sgd_lr;
    float _sgd_lr_min;
    boost::shared_mutex _embedding_mut;
    /*
    std::unordered_map<uint64_t, Eigen::ArrayXf> _word_embeddings;
    std::unordered_map<uint64_t, Eigen::ArrayXf> _context_embeddings;
    std::unordered_map<uint64_t, Eigen::ArrayXf> _word_grad_sq_sum;
    std::unordered_map<uint64_t, Eigen::ArrayXf> _context_grad_sq_sum;
    */
    /*
    tbb::concurrent_hash_map<uint64_t, Eigen::ArrayXf> _word_embeddings;
    tbb::concurrent_hash_map<uint64_t, Eigen::ArrayXf> _context_embeddings;
    tbb::concurrent_hash_map<uint64_t, Eigen::ArrayXf> _word_grad_sq_sum;
    tbb::concurrent_hash_map<uint64_t, Eigen::ArrayXf> _context_grad_sq_sum;
    */
    tbb::concurrent_unordered_map<uint64_t, uint64_t> _context_counts;  // store the not-so-accurate update counts of the context words
    tbb::concurrent_unordered_map<uint64_t, Eigen::ArrayXf> _word_embeddings;
    tbb::concurrent_unordered_map<uint64_t, Eigen::ArrayXf> _context_embeddings;
    tbb::concurrent_unordered_map<uint64_t, Eigen::ArrayXf> _word_grad_sq_sum;
    //tbb::concurrent_unordered_map<uint64_t, Eigen::ArrayXf> _context_grad_sq_sum; // the RMSprop can't be noisy?
    

    size_t _trainer_num;
    std::vector<std::shared_ptr<boost::thread> > _trainer_thrds;

    int64_t _dumper_intv;
    std::shared_ptr<boost::thread> _dumper_timer_thrd;

    int64_t _cur_ts;
    int64_t _expire_ts;
    int64_t _expire_intv;
    std::shared_ptr<boost::thread> _expire_timer_thrd;

    void rnd_init(Eigen::ArrayXf &arr, const size_t dim) {
        uni_dist.reset();
        arr = Eigen::ArrayXf::Zero(dim);
        for (size_t i = 0; i < dim; ++i) {
            arr[i] = uni_dist(rnd_gen);
            if (std::abs(arr[i]) < 0.05) {
                arr[i] = 0.05;
            }
        }
    }

    template <class RndNumDistFunc>
    static void rnd_init(Eigen::ArrayXf &arr, const size_t dim, RndNumDistFunc &uni) {
        arr = Eigen::ArrayXf::Zero(dim);
        for (size_t i = 0; i < dim; ++i) {
            arr[i] = uni();
            if (std::abs(arr[i]) < 0.05) {
                arr[i] = 0.05;
            }
        }
    }

    static void train_thrd(Word2Vec *wv, const size_t idx) {

        std::mt19937_64 rnd_gen(\
            boost::hash<boost::thread::id>()(boost::this_thread::get_id()));
        uint64_t proc_count = 0;
        std::uniform_real_distribution<double> u1_dist(-1, 1);
        auto rnd_num_gen = [&] { return u1_dist(rnd_gen); };

        std::vector<uint64_t> neg_samp_vec(wv->_neg_num);
        Eigen::ArrayXf w_grad_tmp = Eigen::ArrayXf::Zero(wv->_embedding_dim);
        Eigen::ArrayXf c_grad_tmp = Eigen::ArrayXf::Zero(wv->_embedding_dim);
        WordContextPair wcp;
        Eigen::ArrayXf rnd_initer;
        while (true) {

            if (proc_count > 500000) {
                unsigned seed = static_cast<unsigned>(100 * u1_dist(rnd_gen));
                rnd_gen.seed(boost::posix_time::microsec_clock::local_time(\
                    ).time_of_day().total_milliseconds() + seed);
                proc_count = 0;
            }

            // get positive training pair sample
            wv->_word_sampler.get_word_context(wcp, idx);
            // get negative training pair samples
            auto end_it = wv->_word_sampler.get_neg(wv->_neg_num, neg_samp_vec.begin());
            //boost::upgrade_lock<boost::shared_mutex> lock(wv->_embedding_mut);
            boost::shared_lock<boost::shared_mutex> lock(wv->_embedding_mut);   // why this is needed? 'cause we need to expire keys in the future.
            // init word & context embeddings if not found
            if (!wv->_word_embeddings.count(wcp.word)) {
                //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
                wv->_word_grad_sq_sum.emplace(wcp.word, Eigen::ArrayXf::Zero(wv->_embedding_dim));
                rnd_init(rnd_initer, wv->_embedding_dim, rnd_num_gen);
                wv->_word_embeddings.emplace(wcp.word, rnd_initer);
            }
            if (!wv->_context_embeddings.count(wcp.context)) {
                //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
                wv->_context_counts.emplace(wcp.context, 0);
                //wv->_context_grad_sq_sum.emplace(wcp.context, Eigen::ArrayXf::Zero(wv->_embedding_dim));
                rnd_init(rnd_initer, wv->_embedding_dim, rnd_num_gen);
                wv->_context_embeddings.emplace(wcp.context, rnd_initer);
            }
            wv->_context_counts[wcp.context]++;
            for (auto it = neg_samp_vec.begin(); it != end_it; ++it) {
                if (!wv->_context_embeddings.count(*it)) {
                    //boost::upgrade_to_unique_lock<boost::shared_mutex> unique_lock(lock);
                    wv->_context_counts.emplace(*it, 0);
                    //wv->_context_grad_sq_sum.emplace(*it, Eigen::ArrayXf::Zero(wv->_embedding_dim));
                    rnd_init(rnd_initer, wv->_embedding_dim, rnd_num_gen);
                    wv->_context_embeddings.emplace(*it, rnd_initer);
                }
                wv->_context_counts[*it]++;
            }
            // the real training process
            //  the positive sample
            double wc_dot = wv->_word_embeddings[wcp.word].matrix().dot(wv->_context_embeddings[wcp.context].matrix());
            //      the grads
            w_grad_tmp = (-1 + wv->sigtable(wc_dot)) * wv->_context_embeddings[wcp.context];
            //      update context embedding
            c_grad_tmp = (-1 + wv->sigtable(wc_dot)) * wv->_word_embeddings[wcp.word];
            float ctx_lr = wv->_sgd_lr / static_cast<float>(wv->_context_counts[wcp.context] / 4 + 1);
            if (ctx_lr < wv->_sgd_lr_min) {
                ctx_lr = wv->_sgd_lr_min;
            }
            wv->_context_embeddings[wcp.context] -= ctx_lr * c_grad_tmp;
            // the negative
            for (auto neg_iter = neg_samp_vec.begin(); neg_iter != end_it; ++neg_iter) {
                wc_dot = wv->_word_embeddings[wcp.word].matrix().dot(wv->_context_embeddings[*neg_iter].matrix());
                w_grad_tmp += wv->sigtable(wc_dot) * wv->_context_embeddings[*neg_iter];
                c_grad_tmp = wv->sigtable(wc_dot) * wv->_word_embeddings[wcp.word];
                ctx_lr = wv->_sgd_lr / static_cast<float>(wv->_context_counts[*neg_iter] / 4 + 1);
                if (ctx_lr < wv->_sgd_lr_min) {
                    ctx_lr = wv->_sgd_lr_min;
                }
                wv->_context_embeddings[*neg_iter] -= ctx_lr * c_grad_tmp;
            }
            // update the word embedding
            wv->_word_grad_sq_sum[wcp.word] *= wv->_gamma;
            wv->_word_grad_sq_sum[wcp.word] += (1.0 - wv->_gamma) * w_grad_tmp.square();
            wv->_word_embeddings[wcp.word] -= wv->_learning_rate * (w_grad_tmp \
                / (wv->_epsilon + wv->_word_grad_sq_sum[wcp.word]).sqrt());
        }
    }

    static void dumper_timer(Word2Vec *wv) {
        int64_t pre_ts = std::chrono::duration_cast<std::chrono::seconds>(\
            std::chrono::system_clock::now().time_since_epoch()).count();
        while (true) {
            int64_t cur_ts = std::chrono::duration_cast<std::chrono::seconds>(\
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (cur_ts - pre_ts < wv->_dumper_intv) {
                boost::this_thread::sleep(boost::posix_time::seconds(1));
                continue;
            }

            std::string dumper_fn("word2vec.dumper." + std::to_string(pre_ts));
            std::ofstream dumperf(dumper_fn);
            //std::unordered_map<uint64_t, Eigen::ArrayXf> wemb;
            //tbb::concurrent_hash_map<uint64_t, Eigen::ArrayXf, boost::hash<uint64_t>()> wemb;
            tbb::concurrent_unordered_map<uint64_t, Eigen::ArrayXf> wemb;
            {
                //boost::shared_lock<boost::shared_mutex> lock(wv->_embedding_mut);
                wemb = wv->_word_embeddings;
            }
            dumperf << wemb.size() << "\t" << wv->_embedding_dim << std::endl;
            for (const auto &item : wemb) {
                dumperf << item.first << "\t" << item.second.transpose() << std::endl;
            }
            dumperf.close();
            pre_ts = std::chrono::duration_cast<std::chrono::seconds>(\
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
    }

    static void expire_timer(Word2Vec *wv) {
        std::vector<uint64_t> words_del;
        std::vector<uint64_t> docs_del;
        int64_t pre_ts = std::chrono::duration_cast<std::chrono::seconds>(\
            std::chrono::system_clock::now().time_since_epoch()).count();
        while (true) {
            int64_t cur_ts = std::chrono::duration_cast<std::chrono::seconds>(\
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (cur_ts - pre_ts < wv->_expire_intv) {
                boost::this_thread::sleep(boost::posix_time::seconds(100));
                continue;
            }

            wv->_word_sampler.del_expire(wv->_cur_ts - wv->_expire_ts, words_del, docs_del);
            boost::unique_lock<boost::shared_mutex> lock(wv->_embedding_mut);
            for (auto &word : words_del) {
                wv->_context_counts.unsafe_erase(word);
                wv->_word_embeddings.unsafe_erase(word);
                wv->_context_embeddings.unsafe_erase(word);
                wv->_word_grad_sq_sum.unsafe_erase(word);
                //wv->_context_grad_sq_sum.unsafe_erase(word);
            }
            lock.unlock();
            pre_ts = std::chrono::duration_cast<std::chrono::seconds>(\
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
    }
};

}
