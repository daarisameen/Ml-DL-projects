# ML MODELS
Algorithms are often grouped by similarity in terms of their function

# Regression Algorithms

![Alt text](https://machinelearningmastery.com/wp-content/uploads/2013/11/Regression-Algorithms.png)

Regression AlgorithmsRegression is concerned with modeling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model.

Regression methods are a workhorse of statistics and have been co-opted into statistical machine learning. This may be confusing because we can use regression to refer to the class of problem and the class of algorithm. Really, regression is a process.

The most popular regression algorithms are:

Ordinary Least Squares Regression (OLSR)
Linear Regression
Logistic Regression
Stepwise Regression
Multivariate Adaptive Regression Splines (MARS)
Locally Estimated Scatterplot Smoothing (LOESS)

# Instance-based Algorithms

![Alt text](https://machinelearningmastery.com/wp-content/uploads/2013/11/Instance-based-Algorithms.png)

Instance-based AlgorithmsInstance-based learning model is a decision problem with instances or examples of training data that are deemed important or required to the model.

Such methods typically build up a database of example data and compare new data to the database using a similarity measure in order to find the best match and make a prediction. For this reason, instance-based methods are also called winner-take-all methods and memory-based learning. Focus is put on the representation of the stored instances and similarity measures used between instances.

The most popular instance-based algorithms are:

k-Nearest Neighbor (kNN)
Learning Vector Quantization (LVQ)
Self-Organizing Map (SOM)
Locally Weighted Learning (LWL)
Support Vector Machines (SVM)

# Regularization Algorithms

Regularization AlgorithmsAn extension made to another method (typically regression methods) that penalizes models based on their complexity, favoring simpler models that are also better at generalizing.

I have listed regularization algorithms separately here because they are popular, powerful and generally simple modifications made to other methods.

The most popular regularization algorithms are:

Ridge Regression
Least Absolute Shrinkage and Selection Operator (LASSO)
Elastic Net
Least-Angle Regression (LARS)

# Decision Tree Algorithms

Decision Tree AlgorithmsDecision tree methods construct a model of decisions made based on actual values of attributes in the data.

Decisions fork in tree structures until a prediction decision is made for a given record. Decision trees are trained on data for classification and regression problems. Decision trees are often fast and accurate and a big favorite in machine learning.

The most popular decision tree algorithms are:

Classification and Regression Tree (CART)
Iterative Dichotomiser 3 (ID3)
C4.5 and C5.0 (different versions of a powerful approach)
Chi-squared Automatic Interaction Detection (CHAID)
Decision Stump
M5
Conditional Decision Trees

# Bayesian Algorithms

Bayesian AlgorithmsBayesian methods are those that explicitly apply Bayes’ Theorem for problems such as classification and regression.

The most popular Bayesian algorithms are:

Naive Bayes
Gaussian Naive Bayes
Multinomial Naive Bayes
Averaged One-Dependence Estimators (AODE)
Bayesian Belief Network (BBN)
Bayesian Network (BN)

# Clustering Algorithms

Clustering AlgorithmsClustering, like regression, describes the class of problem and the class of methods.

Clustering methods are typically organized by the modeling approaches such as centroid-based and hierarchal. All methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

The most popular clustering algorithms are:

k-Means
k-Medians
Expectation Maximisation (EM)
Hierarchical Clustering
Association Rule Learning Algorithms
Assoication Rule Learning AlgorithmsAssociation rule learning methods extract rules that best explain observed relationships between variables in data.

These rules can discover important and commercially useful associations in large multidimensional datasets that can be exploited by an organization.

The most popular association rule learning algorithms are:

Apriori algorithm
Eclat algorithm
Artificial Neural Network Algorithms
Artificial Neural Network AlgorithmsArtificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks.

They are a class of pattern matching that are commonly used for regression and classification problems but are really an enormous subfield comprised of hundreds of algorithms and variations for all manner of problem types.

Note that I have separated out Deep Learning from neural networks because of the massive growth and popularity in the field. Here we are concerned with the more classical methods.

The most popular artificial neural network algorithms are:

Perceptron
Multilayer Perceptrons (MLP)
Back-Propagation
Stochastic Gradient Descent
Hopfield Network
Radial Basis Function Network (RBFN)
Deep Learning Algorithms
Deep Learning AlgorithmsDeep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation.

They are concerned with building much larger and more complex neural networks and, as commented on above, many methods are concerned with very large datasets of labelled analog data, such as image, text. audio, and video.

The most popular deep learning algorithms are:

Convolutional Neural Network (CNN)
Recurrent Neural Networks (RNNs)
Long Short-Term Memory Networks (LSTMs)
Stacked Auto-Encoders
Deep Boltzmann Machine (DBM)
Deep Belief Networks (DBN)
Dimensionality Reduction Algorithms
Dimensional Reduction AlgorithmsLike clustering methods, dimensionality reduction seek and exploit the inherent structure in the data, but in this case in an unsupervised manner or order to summarize or describe data using less information.

This can be useful to visualize dimensional data or to simplify data which can then be used in a supervised learning method. Many of these methods can be adapted for use in classification and regression.

Principal Component Analysis (PCA)
Principal Component Regression (PCR)
Partial Least Squares Regression (PLSR)
Sammon Mapping
Multidimensional Scaling (MDS)
Projection Pursuit
Linear Discriminant Analysis (LDA)
Mixture Discriminant Analysis (MDA)
Quadratic Discriminant Analysis (QDA)
Flexible Discriminant Analysis (FDA)
t-distributed Stochastic Neighbor Embedding (t-SNE)
Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP)
Ensemble Algorithms
Ensemble AlgorithmsEnsemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction.

Much effort is put into what types of weak learners to combine and the ways in which to combine them. This is a very powerful class of techniques and as such is very popular.

Boosting
Bootstrapped Aggregation (Bagging)
AdaBoost
Weighted Average (Blending)
Stacked Generalization (Stacking)
Gradient Boosting Machines (GBM)
Gradient Boosted Regression Trees (GBRT)
Random Forest
Other Machine Learning Algorithms
Many algorithms were not covered.

I did not cover algorithms from specialty tasks in the process of machine learning, such as:

Feature selection algorithms
Algorithm accuracy evaluation
Performance measures
Optimization algorithms
I also did not cover algorithms from specialty subfields of machine learning, such as:

Computational intelligence (evolutionary algorithms, etc.)
Computer Vision (CV)
Natural Language Processing (NLP)
Recommender Systems
Reinforcement Learning
Graphical Models
And more…
