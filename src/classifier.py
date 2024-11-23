import numpy as np


class SpaceLogisticRegressor:
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None

    def fit(self, X, y, prev_scores):
        _, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=np.float32)

        for _ in range(self.max_iter):
            # Calculate current logits (raw scores)
            logits = prev_scores + X @ self.coef_
            logits = np.clip(logits, -20, 20)  # Adjusted clipping range

            # Convert logits to probabilities
            probabilities = np.exp(-np.logaddexp(0, -logits))

            # Compute gradients (modified logistic gradient)
            gradients = X.T @ (y - probabilities) - self.alpha * self.coef_

            # Compute Hessian approximation
            diag = probabilities * (1 - probabilities)
            hessian = X.T @ (diag[:, np.newaxis] * X) + self.alpha * np.eye(n_features)

            # Add regularization to Hessian for stability
            epsilon = 1e-6
            hessian += epsilon * np.eye(n_features)

            # Update coefficients
            update = np.linalg.solve(hessian, gradients)
            self.coef_ += update

            # Convergence check
            if np.linalg.norm(update) / max(1.0, np.linalg.norm(self.coef_)) < self.tol:
                break

    def predict(self, X, prev_scores):
        # Predict raw scores (logits)
        logits = prev_scores + X @ self.coef_
        return logits  # Raw scores are returned


class SpaceTreeClassifier:
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

    def _find_direction(self, X, y, prev_scores):
        # Use logistic regression to find the optimal direction
        lr = SpaceLogisticRegressor(alpha=self.alpha)

        # Fit logistic regression on the original target `y` and previous logit scores
        lr.fit(X, y, prev_scores)

        # Normalize the resulting coefficient vector to get the direction
        direction = lr.coef_
        return direction / np.linalg.norm(direction)

    def _project_data(self, X):
        return X @ self.direction

    def _find_best_split(self, X, gradients, hessians):
        n_samples = len(gradients)
        if n_samples < 2 * self.min_samples_split:
            return None, None, None

        projections = self._project_data(X)
        sort_idx = np.argsort(projections)
        sorted_projections = projections[sort_idx]
        sorted_gradients = gradients[sort_idx]
        sorted_hessians = hessians[sort_idx]

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

        cumsum_gradients = np.cumsum(sorted_gradients)
        cumsum_hessians = np.cumsum(sorted_hessians)

        left_gradients = cumsum_gradients[split_indices - 1]
        left_hessians = cumsum_hessians[split_indices - 1]

        right_gradients = cumsum_gradients[-1] - left_gradients
        right_hessians = cumsum_hessians[-1] - left_hessians

        left_n = split_indices
        right_n = n_samples - split_indices

        valid_splits = (left_n >= self.min_samples_leaf) & (
            right_n >= self.min_samples_leaf
        )
        if not np.any(valid_splits):
            return None, None, None

        left_loss = -(left_gradients**2) / (left_hessians + 1e-10)
        right_loss = -(right_gradients**2) / (right_hessians + 1e-10)

        total_loss = left_loss + right_loss
        best_idx = np.argmin(total_loss[valid_splits])
        best_threshold = sorted_projections[split_indices[valid_splits][best_idx]]
        best_split_mask = projections <= best_threshold

        return best_threshold, best_split_mask, total_loss[valid_splits][best_idx]

    def _build_tree(self, X, gradients, hessians, depth=0):
        n_samples = len(gradients)
        node = {"n_samples": n_samples}

        # Yaprak düğüm kontrolü
        if depth == self.max_depth or n_samples < 2 * self.min_samples_split:
            node["is_leaf"] = True
            node["value"] = np.sum(gradients) / (
                np.sum(hessians) + 1e-10
            )  # Raw logit value
            return node

        threshold, split_mask, score = self._find_best_split(X, gradients, hessians)

        if threshold is None:
            node["is_leaf"] = True
            node["value"] = np.sum(gradients) / (np.sum(hessians) + 1e-10)
            return node

        X_left, gradients_left, hessians_left = (
            X[split_mask],
            gradients[split_mask],
            hessians[split_mask],
        )
        X_right, gradients_right, hessians_right = (
            X[~split_mask],
            gradients[~split_mask],
            hessians[~split_mask],
        )

        if len(gradients_left) == 0 or len(gradients_right) == 0:
            node["is_leaf"] = True
            node["value"] = np.sum(gradients) / (np.sum(hessians) + 1e-10)
            return node

        node.update(
            {
                "is_leaf": False,
                "threshold": threshold,
                "left": self._build_tree(
                    X_left, gradients_left, hessians_left, depth + 1
                ),
                "right": self._build_tree(
                    X_right, gradients_right, hessians_right, depth + 1
                ),
            }
        )

        return node

    def fit(self, X, y, prev_scores):
        """
        Fit the tree on the gradients and hessians derived from the previous ensemble's predictions.
        X: Feature matrix
        y: Original target (0 or 1)
        prev_scores: Logit scores from previous ensemble predictions
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Compute probabilities and gradients
        logits = prev_scores
        probabilities = np.exp(-np.logaddexp(0, -logits))

        gradients = y - probabilities
        hessians = probabilities * (1 - probabilities)

        # Find projection direction
        self.direction = self._find_direction(X, y, prev_scores)

        # Build the tree
        self.tree = self._build_tree(X, gradients, hessians)
        return self

    def predict_proba(self, X, raw=False):
        X = np.asarray(X, dtype=np.float32)
        logits = np.zeros(len(X), dtype=np.float32)
        nodes = [self.tree] * len(X)
        mask = np.ones(len(X), dtype=bool)

        while np.any(mask):
            leaf_mask = mask & np.array([node["is_leaf"] for node in nodes])
            if np.any(leaf_mask):
                logits[leaf_mask] = np.array(
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

        return logits if raw else np.exp(-np.logaddexp(0, -logits))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)


class SpaceBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        alpha=0.0,
        min_samples_split=2,
        min_samples_leaf=1,
        n_splits=255,
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

        # Initialize raw scores (logit of mean target)
        mean_y = np.mean(y)
        self.init_prediction = np.log(mean_y / (1 - mean_y))
        current_predictions = np.full(
            len(y), self.init_prediction, dtype=np.float32
        )  # Raw scores

        for i in range(self.n_estimators):
            # Train a tree on gradients
            tree = SpaceTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_splits=self.n_splits,
                alpha=self.alpha,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )
            tree.fit(X, y, current_predictions)  # Pass raw scores
            self.trees.append(tree)

            # Update raw scores with tree predictions
            current_predictions += self.learning_rate * tree.predict_proba(X, raw=True)

    def predict_proba(self, X, raw=False):
        logits = np.full(len(X), self.init_prediction, dtype=np.float32)
        for tree in self.trees:
            logits += self.learning_rate * tree.predict_proba(X, raw=True)
        return logits if raw else 1 / (1 + np.exp(-logits))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
