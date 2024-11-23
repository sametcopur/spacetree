import numpy as np
from sklearn.linear_model import Ridge

class DirectionalDecisionTree:
    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        n_splits=255,
        alpha=0.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
        self.tree = None
        self.direction = None
        self.alpha = alpha
        self.random_state = random_state

    def _find_direction(self, X, y):
        lr = Ridge(alpha=self.alpha, random_state=self.random_state)
        lr.fit(X, y)
        direction = lr.coef_
        return direction / np.sqrt(np.sum(direction**2))

    def _project_data(self, X):
        return X @ self.direction

    def _find_best_split(self, X, y):
        n_samples = len(y)

        if n_samples < 2 * self.min_samples_split:
            return None, None, None

        projections = self._project_data(X)
        sort_idx = np.argsort(projections)
        sorted_projections = projections[sort_idx]
        sorted_y = y[sort_idx]

        max_splits = n_samples - 2 * self.min_samples_split
        if max_splits <= 0:
            return None, None, None

        n_splits = min(self.n_splits, max_splits)
        split_indices = np.linspace(
            self.min_samples_split,
            n_samples - self.min_samples_split,
            num=n_splits,
            dtype=int,
        )

        if len(split_indices) == 0:
            return None, None, None

        cumsum_y = np.cumsum(sorted_y)
        sq_cumsum_y = np.cumsum(sorted_y**2)
        total_sum = cumsum_y[-1]
        total_sq_sum = sq_cumsum_y[-1]

        left_sums = cumsum_y[split_indices - 1]
        left_counts = split_indices
        right_counts = n_samples - split_indices
        right_sums = total_sum - left_sums

        left_sq_sums = sq_cumsum_y[split_indices - 1]
        right_sq_sums = total_sq_sum - left_sq_sums

        valid_splits = (left_counts >= self.min_samples_leaf) & (
            right_counts >= self.min_samples_leaf
        )
        if not np.any(valid_splits):
            return None, None, None

        left_mse = np.where(
            left_counts > 0,
            (left_sq_sums - (left_sums**2 / left_counts)) / left_counts,
            np.inf,
        )
        right_mse = np.where(
            right_counts > 0,
            (right_sq_sums - (right_sums**2 / right_counts)) / right_counts,
            np.inf,
        )
        total_mse = (left_counts * left_mse + right_counts * right_mse) / n_samples

        best_idx = np.argmin(total_mse[valid_splits])
        best_threshold = sorted_projections[split_indices[valid_splits][best_idx]]
        best_split_mask = projections <= best_threshold

        return best_threshold, best_split_mask, total_mse[valid_splits][best_idx]

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        node = {"n_samples": n_samples, "value": np.mean(y)}

        if depth == self.max_depth or n_samples < 2 * self.min_samples_split:
            node["is_leaf"] = True
            return node

        threshold, split_mask, score = self._find_best_split(X, y)

        if threshold is None:
            node["is_leaf"] = True
            return node

        X_left, y_left = X[split_mask], y[split_mask]
        X_right, y_right = X[~split_mask], y[~split_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            node["is_leaf"] = True
            return node

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
        if len(y) < 2 * self.min_samples_split:
            raise ValueError(
                f"Not enough samples for min_samples_split={self.min_samples_split}"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.direction = self._find_direction(X, y)
        self.tree = self._build_tree(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        predictions = np.zeros(len(X), dtype=np.float32)
        nodes = [self.tree] * len(X)
        mask = np.ones(len(X), dtype=bool)

        while np.any(mask):
            leaf_mask = mask & np.array([node["is_leaf"] for node in nodes])
            if np.any(leaf_mask):
                predictions[leaf_mask] = np.array(
                    [node["value"] for node in np.array(nodes)[leaf_mask]]
                )
                mask &= ~leaf_mask

            if not np.any(mask):
                break

            projections = self._project_data(X[mask])
            current_nodes = np.array(nodes)[mask]
            thresholds = np.array([node["threshold"] for node in current_nodes])

            go_left = projections <= thresholds

            new_nodes = np.array(nodes)
            new_nodes[mask] = np.where(
                go_left,
                [node["left"] for node in current_nodes],
                [node["right"] for node in current_nodes],
            )
            nodes = new_nodes.tolist()

        return predictions


class GradientBoostingDirectionalTree:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        alpha=0.0,
        min_samples_split=2,
        min_samples_leaf=5,
        n_splits=5,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
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
            tree = DirectionalDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_splits=self.n_splits,
                alpha=self.alpha,
                random_state=(self.random_state + i if self.random_state is not None else None),
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
