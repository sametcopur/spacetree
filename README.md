

## About the Project

This project contains two innovative regression models: **SpaceTreeRegressor** and **SpaceBoostingRegressor**. Unlike traditional decision trees, these models split the data by projecting it along a specific direction. The **SpaceTreeRegressor** splits data in the feature space using a linear projection, while the **SpaceBoostingRegressor** reduces errors iteratively using an ensemble learning approach.

Both models are designed to deliver more efficient and accurate results, particularly with large datasets. You can read more about the mathematical details and how the models work in the **[algorithm documentation here](algo/algo.pdf)**.


To better understand how the **SpaceBoostingRegressor** model works, here is an animation that demonstrates the sequential learning process of adding trees to the ensemble. 

In this animation:
- **Step 1**: The initial prediction starts as the mean of the target values.
- **Step 2**: In each iteration, the residuals are computed (the difference between the actual and predicted values) and a new tree is trained on these residuals.
- **Step 3**: The model is updated by adding the weighted predictions of each tree, with the goal of reducing the overall error.


![SpaceBoostingRegressor Animation](./utils/tree_animation.gif)
