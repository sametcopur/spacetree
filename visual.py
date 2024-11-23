import matplotlib.pyplot as plt
import numpy as np
from spacetree import GradientBoostingDirectionalTree

# Örnek iki boyutlu veri oluştur
np.random.seed(42)
X = np.random.rand(200, 2) * 10  # 200 nokta, 2 özellik

# İki özellik arasında karışık ve doğrusal olmayan bir ilişki
# Define a new mixed relationship
y = (
    (np.sin(X[:, 0]) > 0) * 1.0  # Binary step based on sine
    + (np.cos(X[:, 1]) > 0) * 2.0  # Binary step based on cosine
    + (np.abs(X[:, 0] - X[:, 1]) > 5) * 3.0  # Discontinuity based on distance
    + np.random.choice([0, 1], size=200, p=[0.8, 0.2])  # Random noise (discrete)
)
# DirectionalDecisionTree sınıfını alıyoru
def plot_trees_colormap_with_comparison(X, y, model, interval=0.5, resolution=100):
    """
    Visualize predictions after each tree is added in the gradient boosting model
    using side-by-side comparison of true values and tree predictions.

    Parameters:
    - X: Input features (2D array-like), shape (n_samples, 2).
    - y: True target values (1D array-like), shape (n_samples,).
    - model: Fitted GradientBoostingDirectionalTree instance.
    - interval: Pause duration between plots (in seconds).
    - resolution: Grid resolution for the color map.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    # Create a grid over the input space for predictions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Initialize predictions with the initial mean prediction
    predictions = np.full(len(y), model.init_prediction, dtype=np.float32)
    cumulative_predictions = np.full(len(grid_points), model.init_prediction, dtype=np.float32)
    
    plt.figure(figsize=(16, 8))
    
    # Loop through the trees and plot predictions
    for i, tree in enumerate(model.trees):
        predictions += model.learning_rate * tree.predict(X)
        cumulative_predictions += model.learning_rate * tree.predict(grid_points)

        # Clear the plot for the current iteration
        plt.clf()
        
        # Plot true values on the left
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="coolwarm", s=50, label="True Values")
        plt.title("True Values")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="True Values")
        
        # Plot model predictions on the right
        plt.subplot(1, 2, 2)
        plt.contourf(xx, yy, cumulative_predictions.reshape(xx.shape), alpha=0.8, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=predictions, edgecolor='k', cmap="coolwarm", s=50, label="Tree Predictions")
        plt.title(f"Tree Predictions - Iteration {i + 1}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Prediction")
        
        plt.pause(interval)
    
    # Show the final result
    plt.show()

# Example usage after training the model
gb_model = GradientBoostingDirectionalTree(
    n_estimators=100, learning_rate=0.1, max_depth=6, n_splits=255,
    min_samples_split=2, min_samples_leaf=1, alpha=3, random_state=32
)
gb_model.fit(X, y)
plot_trees_colormap_with_comparison(X, y, gb_model, interval=0.5, resolution=200)
