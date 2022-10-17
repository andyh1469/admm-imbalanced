# ECE236C Project: ADMM for Class-Imbalanced Training of Binary Image Classifiers

## Summary:
In machine learning, class imbalance is a common challenge for many classification tasks. It can arise in many applications, for example in medical diagnosis of rare skin conditions where the vast majority of available training samples come from people who do not have the disease. Lack of available data from one class often leads to the model prioritizing good performance on the majority class during training. As a result, the classifier exhibits especially poor performance on the minority class during test time. This problem is typically addressed by adding regularization terms to the loss function or pre-processing the training data. These measures still utilize common iterative methods such as gradient descent for training and do not involve changes to the optimization algorithm itself. 

This project uses a dual optimization method (ADMM) to solve a large-scale logistic regression problem for binary image classification, given an imbalanced training set. A hard constraint is added to encourage the model to classify the minority labels with the same effectiveness as the majority labels.

Details on the problem formulation and ADMM derivation are provided in [derivation.pdf](https://github.com/andyh1469/admm_imbalanced/blob/main/derivation.pdf).

## References:
[1]	N. Japkowicz and S. Stephen, “The Class Imbalance Problem: A Systematic Study,” Intelligent Data Analysis, pp. 429-449, 2002.

[2]	S.P. Boyd and L. Vandenberghe, Convex Optimization. Cambridge: Cambridge University Press, 2009.

[3]	S.P. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. “Distributed optimization and statistical learning via the alternating direction method of multipliers”. Foundations and Trends in Machine Learning, 3(1):1122, 2011. 
