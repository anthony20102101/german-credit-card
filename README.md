# German credit card classification

The goal of the dataset is to be able to classify the customers into two different classes. You get 20 data points per customers.  

Both "standard" Ml algorithms and neural networks are tested in the code. The results shows that the neural neural networks outperform the other techniques (Important to note however is that I am more familiar with optimizing neural nets that the other algorithms). Results can be found below. Deeper and wider neural networks were both tried out, but neither outperformed the model that is found in the code. Dropout was also added for regularization however had a negative effect on the score

| Model          | Accuracy |
|----------------|----------|
| SVM            | 70%      |
| Random Forest  | 71%      |
| Neural Network | 78%      |

This project was done purely because it is fun and to increase my familiarity with the pytorch framework.
