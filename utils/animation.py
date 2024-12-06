import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from src.regressor import SpaceBoostingRegressor

# Örnek iki boyutlu veri oluştur
np.random.seed(42)
X = np.random.rand(200, 2) * 10  # 200 nokta, 2 özellik

# İki özellik arasında karışık ve doğrusal olmayan bir ilişki
y = (
    (np.sin(X[:, 0]) > 0) * 1.0
    + (np.cos(X[:, 1]) > 0) * 2.0
    + (np.abs(X[:, 0] - X[:, 1]) > 5) * 3.0
    + np.random.choice([0, 1], size=200, p=[0.8, 0.2])
)

def plot_trees_animation(X, y, model, resolution=200, filename="animation.mp4"):
    """
    Creates an animation of tree predictions and saves it to a file.
    
    Parameters:
    - X: Input features (2D array-like), shape (n_samples, 2).
    - y: True target values (1D array-like), shape (n_samples,).
    - model: Fitted GradientBoostingDirectionalTree instance.
    - resolution: Grid resolution for the color map.
    - filename: Output file name for the animation.
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
    cumulative_predictions = np.full(len(grid_points), model.init_prediction, dtype=np.float32)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    def update(frame):
        """Update function for the animation."""
        tree = model.trees[frame]
        axes[0].clear()
        axes[1].clear()

        # Update cumulative predictions
        cumulative_predictions[:] += model.learning_rate * tree.predict(grid_points)
        
        # Plot true values
        axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="coolwarm", s=50)
        axes[0].set_title("True Values")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        
        # Plot model predictions
        axes[1].contourf(xx, yy, cumulative_predictions.reshape(xx.shape), alpha=0.8, cmap="coolwarm")
        axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="coolwarm", s=50)
        axes[1].set_title(f"Tree Predictions - Iteration {frame + 1}")
        axes[1].set_xlabel("Feature 1")
        axes[1].set_ylabel("Feature 2")
        
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(model.trees), interval=100)
    
    # Save animation as MP4
    anim.save(filename, writer='ffmpeg', fps=30)
    plt.close(fig)

# Example usage after training the model
gb_model = SpaceBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    min_samples_split=2, min_samples_leaf=1, alpha=3, random_state=32
)
gb_model.fit(X, y)

# Create and save the animation
plot_trees_animation(X, y, gb_model, resolution=200, filename="tree_animation.gif")
