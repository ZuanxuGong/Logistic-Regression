# Logistic-Regression
Write and train a Logistic regression using Gradient Descent for Hand-written digit classification (MNIST).

## Data
Data contains 6 files: <br>
1) trainingData.txt <br> 
2) trainingLabels.txt <br> 
3) validationData.txt <br> 
4) validationLabels.txt <br> 
5) testData.txt <br> 
6) testLabels.txt <br>

## Data format
### “*Data.txt” files contain features.
Each row is comma-separated 784 integers which are features of a single sample. <br>
The number of rows (=number of samples) for training, validation, and test data are 6107, 6107 and 2037 respectively. <br>
### “*Labels.txt” files contain +1 or -1 labels. 
Each row corresponds to the corresponding row of the feature file. 
 
## Steps
- Implement the Gradient Descent algorithm in plain numpy. The core of Gradient Descent is simply the following line:
𝑤(𝑡 + 1) ← 𝑤(𝑡) − 𝜂 𝛻𝑓(𝑤(𝑡)) where 𝑓(𝑤(𝑡)) = 1/𝑁 * ∑𝑖 𝐿(𝑥𝑖, 𝑦𝑖; 𝑤) is the empirical risk.
- Assume initial w is all zeros, the total number of iterations T is 1000, and learning rate 𝜂 is a constant.
- Cross-validation. Training with different values of μ = {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3, 3, 10}.
- The best value 𝜂 = 1.
- The test error is 0.119522. <br> 
PS: remember to change the path.
