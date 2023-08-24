# ADMM for Class-Imbalanced Training of Binary Image Classifiers

# Summary:
In machine learning, class imbalance is a common challenge for many classification tasks. It can arise in many applications, for example in medical diagnosis of rare skin conditions where the vast majority of available training samples come from people who do not have the disease. Lack of available data from one class often leads to the model prioritizing good performance on the majority class during training. As a result, the classifier exhibits especially poor performance on the minority class during test time. This problem is typically addressed by adding regularization terms to the loss function or pre-processing the training data. These measures still utilize common iterative methods such as gradient descent for training and do not involve changes to the optimization algorithm itself. 

This project uses a dual optimization method (ADMM) to solve a large-scale logistic regression problem for binary image classification, given an imbalanced training set. A hard constraint is added to encourage the model to classify the minority labels with the same effectiveness as the majority labels.

# Derivation:

## Dataset and Initial Problem Setup
The dataset used in this project is CIFAR-10, a collection of 32x32 color images from 10 different classes. The coloring model is RGB, bringing the total dimensionality of each image vector to $R^{3072}$. The pixel values were normalized before training, and two of the image classes were selected for the binary classification. The basic training problem for the logistic regression classifier is as follows:

$$ \underset{w}{min} \sum_i^{N_{total}}{log(1+exp(-y_i w^T x_i))} $$

$w \in R^{3073}$ are the weights of the classifier (including bias term), $x_i \in R^{3073}$ is the data vector for the $i^{th}$ image, and $y_i \in \{-1,1\}$ represents the class label of the $i^{th}$ image.

## Adding Convex Constraints
For convenience, the image vectors are concatenated into matrices:

$$ X \in R^{N_{total} x 3073}, X_{major} \in R^{N_{major} x 3073}, X_{minor} \in R^{N_{minor} x 3073} $$

The constrained training approach adds two affine constraints. The first constraint is a change of variables to simplify the ADMM derivation. The second constraint states that the average classifier score $y_i w^T x_i$ of samples from the minority class must be greater than or equal to the average classifier score of samples from the majority class:

$$ \underset{u,w,z}{min} \space\delta_c(z) + \sum_i^{N_{total}}{log(1+exp(-y_i u_i))} $$

$$ subject \space to: $$

$$ Xw - u = 0 $$

$$ m^T w - z = 0 $$

$$ where \space m = -\frac{1}{N_{minor}} X_{minor}^T \textbf{1} - \frac{1}{N_{major}} X_{major}^T \textbf{1}$$

$$ C = \{z|z \geq 0\} $$

## ADMM Derivation
The ADMM updates are derived using $a$ as the dual variable for the first constraint and $b$ as the dual variable for the second constraint. First, $w$ is updated:

$$ w^{(k+1)} = \underset{\hat{w}}{argmin} \space ({a^{(k)}}^T X \hat{w} + \frac{t}{2} {||X \hat{w} - u^{(k)}||}_{2}^{2} + \frac{t}{2} ||m^T \hat{w} - z^{(k)}||_{2}^{2}) $$

$$ = \underset{\hat{w}}{argmin} \space ({a^{(k)}}^T X \hat{w} + \frac{t}{2} (\hat{w}^T X^T X \hat{w} - 2 {\hat{u}^{(k)}}^T X \hat{w}) + \frac{t}{2} ({(m^T \hat{w})}^2 - 2 z^{(k)} m^T \hat{w})) $$

This is a quadratic equation in $\hat{w}$, therefore the $w$ update can be computed by setting the derivative to zero and finding a solution to a system of linear equations:

$$ w^{(k+1)} = (tX^T X + tmm^T)^{-1}q $$ 

$$ where \space q = -X^T a^{(k)} + tX^T u^{(k)} + tz^{(k)} m $$

Next, the variables $u,z$ are jointly updated. The augmented Lagrangian of $u$ is separable:

$$ {u}_i^{(k+1)} = \underset{\hat{u}_i}{argmin} \space (-a_i^{(k)} \hat{u}_i + \frac{t}{2} \hat{u}_i^2 - t(Xw^{(k+1)})_i \hat{u}_i + log(1 + exp(-y_i \hat{u}_i))) $$

$$ i = 1,...,N_{total} $$

Each of the functions in $\hat{u}_i$ are differentiable and can be minimized using an iterative procedure such as Newton's method.

Now the $z$ update is derived:

$$ z^{(k+1)} = \underset{\hat{z}}{argmin} \space (\delta_C(\hat{z}) - b^{(k)} \hat{z} + \frac{t}{2} {||\hat{z} - m^T w^{(k+1)}||}_{2}^{2}) $$

$$ = \underset{\hat{z}}{argmin} \space (\delta_C(\hat{z}) + \frac{t}{2} {||\hat{z} - m^T w^{(k+1)} - \frac{1}{t} b^{(k)}||}_{2}^{2}) $$

$$ = prox_{\frac{1}{t}h} (m^T w^{(k+1)} + \frac{1}{t} b^{(k)}) $$ 

The projection onto $C$ is easy to compute as it is the projection onto the non-negative orthant. Finally, the dual variables are updated:

$$ a^{(k+1)} = a^{(k)} + t(Xw^{(k+1)} - u^{(k+1)}) $$

$$ b^{(k+1)} = b^{(k)} + t(m^T w^{(k+1)} - z^{(k+1)}) $$

# References:
[1]	N. Japkowicz and S. Stephen, “The Class Imbalance Problem: A Systematic Study,” Intelligent Data Analysis, pp. 429-449, 2002.

[2]	S.P. Boyd and L. Vandenberghe, Convex Optimization. Cambridge: Cambridge University Press, 2009.

[3]	S.P. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. “Distributed optimization and statistical learning via the alternating direction method of multipliers”. Foundations and Trends in Machine Learning, 3(1):1122, 2011. 
