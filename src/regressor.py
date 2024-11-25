import numpy as np
from sklearn.linear_model import Ridge


class SpaceTreeRegressor:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        alpha=0.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.direction = None
        self.alpha = alpha
        self.random_state = random_state

    def _find_direction(self, X, y):
        lr = Ridge(alpha=self.alpha, random_state=self.random_state)
        lr.fit(X, y)
        direction = lr.coef_
        return direction / np.sqrt(np.sum(direction ** 2))

    def _project_data(self, X):
        return X @ self.direction

    def _find_best_split(self, X, y):
        n_samples = len(y)

        if n_samples < 2 * self.min_samples_leaf:
            return None, None, None

        projections = self._project_data(X)
        sort_idx = np.argsort(projections)
        sorted_projections = projections[sort_idx]
        sorted_y = y[sort_idx]

        # Compute cumulative sums
        cumsum_y = np.cumsum(sorted_y)
        cumsum_y2 = np.cumsum(sorted_y ** 2)

        total_sum = cumsum_y[-1]
        total_sq_sum = cumsum_y2[-1]

        # Possible split positions
        possible_split_positions = np.arange(
            self.min_samples_leaf, n_samples - self.min_samples_leaf
        )

        # **Add this check**
        if len(possible_split_positions) == 0:
            return None, None, None

        # Left node statistics
        left_counts = possible_split_positions
        left_sums = cumsum_y[possible_split_positions - 1]
        left_sq_sums = cumsum_y2[possible_split_positions - 1]

        # Right node statistics
        right_counts = n_samples - left_counts
        right_sums = total_sum - left_sums
        right_sq_sums = total_sq_sum - left_sq_sums

        # Compute left and right MSE
        left_means = left_sums / left_counts
        right_means = right_sums / right_counts

        left_mse = (
            left_sq_sums - 2 * left_means * left_sums + left_counts * left_means ** 2
        ) / left_counts
        right_mse = (
            right_sq_sums - 2 * right_means * right_sums + right_counts * right_means ** 2
        ) / right_counts

        # Total MSE
        total_mse = (left_counts * left_mse + right_counts * right_mse) / n_samples

        # **Check if total_mse is empty**
        if len(total_mse) == 0:
            return None, None, None

        # Find the best split
        best_idx = np.argmin(total_mse)
        best_mse = total_mse[best_idx]
        threshold = (
            sorted_projections[possible_split_positions[best_idx] - 1]
            + sorted_projections[possible_split_positions[best_idx]]
        ) / 2.0

        split_mask = projections <= threshold

        return threshold, split_mask, best_mse


    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        node = {"n_samples": n_samples, "value": np.mean(y)}

        if depth == self.max_depth or n_samples < 2 * self.min_samples_leaf:
            node["is_leaf"] = True
            return node

        threshold, split_mask, score = self._find_best_split(X, y)

        if threshold is None or np.all(split_mask) or not np.any(split_mask):
            node["is_leaf"] = True
            return node

        X_left, y_left = X[split_mask], y[split_mask]
        X_right, y_right = X[~split_mask], y[~split_mask]

        node.update(
            {
                "is_leaf": False,
                "threshold": threshold,
                "left": self._build_tree(X_left, y_left, depth + 1),
                "right": self._build_tree(X_right, y_right, depth + 1),
            }
        )

        return node

    def fit(self, X, y):
        if len(y) < 2 * self.min_samples_leaf:
            raise ValueError(
                f"Not enough samples for min_samples_leaf={self.min_samples_leaf}"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.direction = self._find_direction(X, y)
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        projections = self._project_data(X)
        predictions = np.zeros(len(X), dtype=np.float32)
        indices = np.arange(len(X))
        node_indices = np.full(len(X), fill_value=-1, dtype=int)
        stack = [(self.tree, indices)]

        while stack:
            node, idx = stack.pop()
            if node["is_leaf"]:
                predictions[idx] = node["value"]
            else:
                threshold = node["threshold"]
                left_idx = idx[projections[idx] <= threshold]
                right_idx = idx[projections[idx] > threshold]

                if len(left_idx) > 0:
                    stack.append((node["left"], left_idx))
                if len(right_idx) > 0:
                    stack.append((node["right"], right_idx))

        return predictions


class SpaceBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        alpha=0.0,
        min_samples_split=2,
        min_samples_leaf=5,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.alpha = alpha
        self.init_prediction = None
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.init_prediction = np.mean(y)
        current_predictions = np.full(len(y), self.init_prediction, dtype=np.float32)

        for i in range(self.n_estimators):
            residuals = y - current_predictions
            tree = SpaceTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                alpha=self.alpha,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            current_predictions += self.learning_rate * tree.predict(X)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        predictions = np.full(len(X), self.init_prediction, dtype=np.float32)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
