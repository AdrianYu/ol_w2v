# ol_w2v
online word2vec for news recommendation.

# what is this
This is the online version of google word2vec model for news recommendation. Word2vec has been proven to successfully capature semantic relations between words dispite of its simple formation. The intuition behind this algorithm is that **words that appear in similar contexts have similar meanings** (Actually both count model, e.g. glove model, and predict model agree this assumption), which lead me to think: what if we **treat users as documents and their news clicks as words**, then will we get a sound model to obtain news embeddings that also capture some latent relations between news? The answer is yes and surprisingly well! (I also tested the glove model, but it performed considerably worse. Part of the reason is that the glove model is much more complicated and has more parameters. (all the decay weights, the positional bias, etc. I just don't exploit all the configurations. Clearly word2vec is a win here.) Before checking my implementation, I highly recommend you to try the original word2vec. Prepare the data, feed into the model, and hours later you will be surprised as I was.

# things need to be addressed
By now, I would assume that you've already tried the original word2vec on your news recommendation logs and find it promissing. But wait, the original word2vec read all the data at once, fix the vocabulary, then do the training. In news recommendation, clearly these conditions cann't meet. Every second, new news rushes in and old news fades out. The vocabulary changes constantly. And time is crucial in news recommendation. The offline routine which performs very good is useless under this scenario. We need a online algorithm! (The gensim lib implements a mini-batch algorithm which is also worth checking)

# the implementation
This project implements the skip-gram with nagetive sampling model in an online fashion. The model itself is quit clear, so I won't address here. The word embeddings are trained using RMSProp and the context embeddings are trained using SGD. (Why is that? First I use RMSProp to train both embeddings which does not work. The reason is that although I can dispatch words to specific trainer, the context words should be globally visible. I must avoid using locks as best as I can. So the gradients would be too noisy for the context words where RMSProp preforms badly.)

This project only implements the online word2vec and is only a draft for now. It is not well documented if not at all, all configurations are hard-coded in the codes and it lacks proper logging. It is fast though. It can process billions of clicks in hours. To use it in production system however, much work still need to be done. Hopefully I will have the time to complete it but I cann't make any promises. Finger crossed.

# dependencies
* [boost](http://www.boost.org/)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [TBB (sorry AMD users)](https://www.threadingbuildingblocks.org/)


