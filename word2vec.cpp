#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "word2vec.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " click_file" << std::endl;
        return 1;
    }

    std::ifstream dataf(argv[1]);
    adrianyu::Word2Vec w2v;
    w2v.input(dataf);

    w2v.join();

    return 0;
}











