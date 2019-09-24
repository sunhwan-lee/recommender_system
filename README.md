# Recommender System using TensorFlow

This repository has the collection of papers and their Python code using Tensorflow related with personalized recommender system.

The repository is continuously updated with more papers and codes. 

## Papers and code

### Matrix Facotrization
* Y. Hu, Y. Koren, and C. Volinsky *Collaborative filtering for implicit feedback datasets*, ICDM, 2008 [[PDF](http://yifanhu.net/PUB/cf.pdf)][code]
* R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. M. Lukose, M. Scholz, and Q. Yang *One-class collaborative filtering*, ICDM, 2008 [[PDF](http://www.rongpan.net/publications/pan-oneclasscf.pdf)][code]
* S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme *BPR: bayesian personalized ranking from implicit feedback*, UAI, 2009 [[PDF](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)][code]
* Y. Koren, R. Bell, and C. Volinsky *Matrix factorization techniques for recommender systems*, Computer, 2009 [[PDF](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)][code]
* Y. Koren, R. Bell *Advances in collaborative filtering*, In Recommender Systems Handbook, 2011 [[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.269.6294&rep=rep1&type=pdf)][code]

### Textual information with explicit signals
Model that incorporate textual feedback to predict star ratings
* J. McAuley, J. Leskovec *Hidden factors and hidden topics: understanding rating dimensions with review text*, RecSys, 2013 [[PDF](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf)][code]
* Q. Diao, M. Qiu, C.-Y. Wu, A. J. Smola, J. Jiang, and C. Wang *Jointly modeling aspects, ratings and sentiments for movie recommendation (JMARS)*, SIGKDD, 2014 [[PDF](https://www.cs.utexas.edu/~cywu/jmars_kdd2014.pdf)][[code](https://github.com/nihalb/JMARS)]
* G. Ling, M. R. Lyu, and I. King *Ratings meet reviews, a combined approach to recommend*, RecSys, 2014 [[PDF](https://dl.acm.org/citation.cfm?id=2645728)][code]
* Y. Wu and M. Ester *FLAME: A probabilistic model combining aspect based opinion mining and collaborative filtering*, WSDM, 2015 [[PDF](https://dl.acm.org/citation.cfm?id=2685291)][code]
* K. Bauman, B. Liu, and A. Tuzhilin *Aspect based recommendations: Recommending items with the most valuable aspects based on user reviews*, SIGKDD, 2017 [[PDF](https://dl.acm.org/citation.cfm?id=3098170)][code]

### Symmetric and asymmetric information with implicit signals
* C. Wang and D. M. Blei *Collaborative topic modeling for recommending scientific articles*, SIGKDD, 2011 [[PDF](http://www.cs.columbia.edu/~blei/papers/WangBlei2011.pdf)][code]
* T. Zhao, J. McAuley, and I. King *Leveraging social connections to improve personalized ranking for collaborative filtering*, CIKM, 2014 [[PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm14.pdf)][code]
* W. Yao, J. He, H. Wang, Y. Zhang, and J. Cao *Collaborative topic ranking: Leveraging item metadata for sparsity reduction*, AAAI, 2015 [[PDF](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9288/9269)][code]
* R. He and J. McAuley *VBPR: Visual bayesian personalized ranking from implicit feedback*, AAAI, 2016 [[PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/aaai16.pdf)][code]

### Knowledge graph embedding
* W. Kang, M. Wan, and J. McAuley *Recommendation Through Mixtures of Heterogeneous Item Relationships*, CIKM, 2018 [[PDF](https://arxiv.org/pdf/1808.10031.pdf)][code]

### Conversational recommender system
* K. Christakopoulou, F. Radlinski, and K. Hofmann *Towards Conversational Recommender Systems*, SIGKDD, 2016 [[PDF](https://www.kdd.org/kdd2016/papers/files/rfp0063-christakopoulouA.pdf)][code]

### Sequential recommendation
* R. He, W. Kang, and J. McAuley *Translation-based recommendation*, RecSys, 2017 [[PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys17.pdf)]

* W. Kang, and J. McAuley *Self-attentive sequential recommendation*, ICDM, 2018 [[PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)]

### Survey paper
* S. Zhang, L. Yao, and A. Sun *Deep learning based recommender system: A survey and new perspectives*, arXiv, 2017 [[PDF](https://arxiv.org/abs/1707.07435)]

## Dataset

* A collection of datasets for recommender systems research from prof. Julian McAuley's lab [[link](https://cseweb.ucsd.edu/~jmcauley/datasets.html)]
  * **Amazon** product reviews and metadata
  * **Amazon** question/answer data
  * **Google Local** business reviews and metadata
  * **Steam** video game reviews and bundles
  * **Goodreads** book reviews
  * **ModCloth** clothing fit feedback
  * **RentTheRunway** clothing fit feedback
  * **Tradesy** bartering data
  * **RateBeer** bartering data
  * **Gameswap** bartering data
  * **Behance** community art reviews and image features
  * **Librarything** reviews and social data
  * **Epinions** reviews and social data
  * **Dance Dance Revolution** step charts
  * **NES** song data
  * **BeerAdvocate** multi-aspect beer reviews
  * **RateBeer** multi-aspect beer reviews
  * **Facebook** social circles data
  * **Twitter** social circles data
  * **Google+** social circles data
  * **Reddit** submission popularity and metadata

* MovieLens dataset (100k, 1M, 20M)
  ```
  curl -O 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
  curl -O 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
  curl -O 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
  ```

* [Last.fm datasets](https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/)