<a name="readme-top"></a>

# 
1. Load the Iris dataset (from sklearn).

2. Split it into training and test sets.

3. Build a 3-layer neural network:

    - Input layer: 4 features
      
    - Hidden layer 1: 8 neurons (ReLU activation)
   
    - Hidden layer 2: 8 neurons (ReLU activation)
      
    - Output layer: 3 neurons (softmax activation)

4. Train the network from scratch using gradient descent.

5. Print the training loss and test accuracy.


How This Code Works

1. Data Loading: We use the Iris dataset from sklearn. It has 150 samples, each with 4 features, and 3 class labels.

2. One-Hot Encoding: Convert class labels (0,1,2) into one-hot vectors (e.g., class 1 → [0,1,0]).
   
3. Train/Test Split: We split the data into 80% training and 20% test.

4. Neural Network Architecture:
   
        - Layer 1: z1 = X W1 + b1, then ReLU → a1.
   
        - Layer 2: z2 = a1 W2 + b2, then ReLU → a2.
   
        - Output Layer: z3 = a2 W3 + b3, then softmax → a3.
   
5. Loss: We use cross-entropy, a common choice for multi-class classification.

6. Backward Pass: Uses the chain rule to compute gradients w.r.t. each weight matrix and bias vector.
   
7. Weight Update: We do a simple gradient descent step on each parameter.

8. Evaluation: We measure test set accuracy by comparing the predicted class index with the true class index.

--------------------------------------------------------------------------------------------------------------------------
== We're Using GitHub Under Protest ==

This project is currently hosted on GitHub.  This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Souce Software
(FOSS).  We are deeply concerned about using a proprietary system like GitHub
to develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term.  We urge you to read about the [Give up GitHub](https://GiveUpGitHub.org) campaign 
from [the Software Freedom Conservancy](https://sfconservancy.org) to understand some of the reasons why GitHub is not 
a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without
using GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission.  We do not consent to GitHub's use of this project's
code in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
