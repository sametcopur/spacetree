# **SpaceTreeRegressor and SpaceBoostingRegressor: Mathematical and Algorithmic Structure**

These models provide innovative solutions to regression problems, with **SpaceTreeRegressor** using projection-based splitting logic and **SpaceBoostingRegressor** utilizing an ensemble learning approach. The mathematical details outline how the data is processed and how the split decisions are made.

---

## **1. SpaceTreeRegressor**

### **How is the Projection Direction Determined?**

`SpaceTreeRegressor`, unlike traditional decision trees, projects data along a specific direction for each split. This direction is computed as follows:

1. **Linear Regression**:
   - Given the data \( X \) (features matrix) and \( y \) (target values), a linear model that best fits the data is found:
     <img src="https://latex.codecogs.com/gif.latex?%5Chat%7By%7D%20%3D%20X%20%5Ccdot%20w" />

2. **Data Projection**:
   - Each data point \( x_i \) is projected along the direction given by the coefficient vector \( w \):
     <img src="https://latex.codecogs.com/gif.latex?p_i%20%3D%20x_i%20%5Ccdot%20w" />
   - This projection maps the high-dimensional data space to a one-dimensional projection axis.

---

### **How is the Split Point Determined?**

After projecting the data, the split point is optimized as follows:

1. **Sorting Projections**:
   - All projection values \( \{p_1, p_2, \dots, p_n\} \) are sorted in ascending order.

2. **Split Candidates**:
   - The sorted projections are used to generate candidate split positions. Possible split positions are selected by ensuring that the resulting left and right groups satisfy the minimum samples per leaf constraint. These split positions correspond to points between adjacent values in the sorted projections.

3. **Error Calculation (MSE)**:
   - For each split candidate, the data is divided into two groups: left (\( L \)) and right (\( R \)).
   - The mean squared error (MSE) for each group is computed:
     <img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BMSE%7D_L%20%3D%20%5Cfrac%7B1%7D%7B%7CL%7C%7D%20%5Csum%5Fi%20%5Cin%20L%20(y_i%20-%20%5Cbar%7By%7D_L)%5E2" />
     <img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BMSE%7D_R%20%3D%20%5Cfrac%7B1%7D%7B%7CR%7C%7D%20%5Csum%5Fi%20%5Cin%20R%20(y_i%20-%20%5Cbar%7By%7D_R)%5E2" />
   - The total MSE is the weighted average of the two groups:
     <img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BTotal%20MSE%7D%20%3D%20%5Cfrac%7B%7CL%7C%7D%7Bn%7D%20%5Ctext%7BMSE%7D_L%20%2B%20%5Cfrac%7B%7CR%7C%7D%7Bn%7D%20%5Ctext%7BMSE%7D_R" />

4. **Selecting the Best Split**:
   - The split candidate that minimizes the total MSE is selected as the optimal threshold \( t^* \).

---

### **How is the Tree Built?**

1. **Data Split**:
   - Based on the chosen threshold \( t^* \), the data is divided into two subgroups:
     <img src="https://latex.codecogs.com/gif.latex?L%20%3D%20%5Cleft%20%7Bi%20%7C%20p_i%20%5Cleq%20t%5E%2A%20%5Cright%20%7D%20%5Cquad%20%5Ctext%20and%20%5Cquad%20R%20%3D%20%5Cleft%20%7Bi%20%7C%20p_i%20%3E%20t%5E%2A%20%5Cright%20%7D" />

2. **Recursion**:
   - The same process is recursively applied to both subgroups.
   - The process stops when the maximum depth is reached, or when the subgroup contains fewer than the minimum number of samples required to split further.

---

## **2. SpaceBoostingRegressor**

### **Boosting Logic**

`SpaceBoostingRegressor` is an ensemble model that sequentially uses multiple `SpaceTreeRegressor` models, aiming to reduce the error at each step. The key here is that the residuals are recalculated in each iteration, causing the projection direction to change with every new tree.

1. **Initial Model**:
   - The first prediction is the mean of the target values \( \bar{y} \):
     <img src="https://latex.codecogs.com/gif.latex?f_0(x)%20%3D%20%5Cbar%7By%7D" />

2. **Residual Learning**:
   - In each iteration, the residuals (errors) are computed based on the current model’s predictions:
     <img src="https://latex.codecogs.com/gif.latex?r_i%20%3D%20y_i%20-%20f_t(x_i)" />
   - These residuals become the new targets for the next iteration. A new `SpaceTreeRegressor` is trained on these residuals to learn the errors that the current model hasn't captured.

3. **Dynamic Projection Direction**:
   - In each boosting iteration, the residuals \( r_i \) are projected using linear regression to determine the new projection direction. This means that the projection direction \( w_t \) is different for each boosting step because the residuals change after each model update:
     <img src="https://latex.codecogs.com/gif.latex?p_i%20%3D%20x_i%20%5Ccdot%20w_t" />
   - As a result, the linear regression coefficients \( w_t \) and the resulting projection direction change in each iteration.

4. **Model Update**:
   - The new model \( h_t(x) \) (the tree built on the residuals) is added to the current model, weighted by the learning rate \( \eta \):
     <img src="https://latex.codecogs.com/gif.latex?f_%7Bt%2B1%7D(x)%20%3D%20f_t(x)%20%2B%20%5Ceta%20h_t(x)" />
     where \( h_t(x) \) is the prediction of the `SpaceTreeRegressor` for the residuals.

5. **Iteration**:
   - The process is repeated for a predefined number of iterations or until the error drops below a specified threshold.

---

### **Final Model and Prediction**

- After training for \( n \) iterations, the final model is a weighted sum of all individual trees:
  <img src="https://latex.codecogs.com/gif.latex?f(x)%20%3D%20f_0(x)%20%2B%20%5Ceta%20%5Csum_%7Bt%3D1%7D%5En%20h_t(x)" />
- The prediction for a new input \( x \) is computed as:
  <img src="https://latex.codecogs.com/gif.latex?%5Chat%7By%7D(x)%20%3D%20f(x)" />

---

To better understand how the **SpaceBoostingRegressor** model works, here is an animation that demonstrates the sequential learning process of adding trees to the ensemble. Each tree learns from the residuals of the previous iteration, adjusting the model to minimize error.

In this animation:
- **Step 1**: The initial prediction starts as the mean of the target values.
- **Step 2**: In each iteration, the residuals are computed (the difference between the actual and predicted values) and a new tree is trained on these residuals.
- **Step 3**: The model is updated by adding the weighted predictions of each tree, with the goal of reducing the overall error.

This process continues iteratively, leading to a progressively more accurate model.

![SpaceBoostingRegressor Animation](./utils/tree_animation.gif)
