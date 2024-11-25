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
        epsilon = 1e-6
        eye = np.eye(n_features)
        hessian_stability = epsilon * eye
        reg = self.alpha * eye

        for _ in range(self.max_iter):
            # Calculate current logits (raw scores)
            logits = prev_scores + X @ self.coef_
            logits = np.clip(logits, -20, 20)  # Adjusted clipping range

            # Convert logits to probabilities
            probabilities = 1 / (1 + np.exp(-logits))

            # Compute gradients (modified logistic gradient)
            gradients = X.T @ (y - probabilities) - self.alpha * self.coef_

            # Compute Hessian approximation
            diag = probabilities * (1 - probabilities)
            hessian = X.T @ (diag[:, np.newaxis] * X) + reg

            # Add regularization to Hessian for stability
            hessian += hessian_stability

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
        alpha=0.0,
        min_gain_split=0.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.direction = None
        self.alpha = alpha
        self.min_gain_split = min_gain_split
        self.random_state = random_state

    def _find_direction(self, X, y, prev_scores):
        # Use logistic regression to find the optimal direction
        lr = SpaceLogisticRegressor(alpha=self.alpha)
        lr.fit(X, y, prev_scores)

        # Normalize the resulting coefficient vector to get the direction
        direction = lr.coef_
        return direction / np.linalg.norm(direction)

    def _project_data(self, X):
        return X @ self.direction

    def _find_best_split(self, start, end):
        n_samples = end - start

        if n_samples < 2 * self.min_samples_leaf:
            return None, None, None

        # Sliced arrays for the current node
        sorted_projections = self.sorted_projections[start:end]

        # Compute cumulative sums for the current node
        cumsum_gradients = self.cumsum_gradients[start:end] - (
            self.cumsum_gradients[start - 1] if start > 0 else 0
        )
        cumsum_hessians = self.cumsum_hessians[start:end] - (
            self.cumsum_hessians[start - 1] if start > 0 else 0
        )

        total_gradients = cumsum_gradients[-1]
        total_hessians = cumsum_hessians[-1]

        # Compute parent loss
        parent_loss = -(total_gradients**2) / (total_hessians + 1e-10)

        # Possible split positions
        possible_split_positions = np.arange(
            self.min_samples_leaf, n_samples - self.min_samples_leaf
        )

        if len(possible_split_positions) == 0:
            return None, None, None

        # Left node statistics
        left_counts = possible_split_positions
        left_gradients = cumsum_gradients[possible_split_positions - 1]
        left_hessians = cumsum_hessians[possible_split_positions - 1]

        # Right node statistics
        right_counts = n_samples - left_counts
        right_gradients = total_gradients - left_gradients
        right_hessians = total_hessians - left_hessians

        # Compute losses
        left_loss = -(left_gradients**2) / (left_hessians + 1e-10)
        right_loss = -(right_gradients**2) / (right_hessians + 1e-10)

        # Total loss after split
        total_loss = left_loss + right_loss

        # Compute gain
        gains = parent_loss - total_loss

        # Find the best split
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]

        # Check if best_gain is greater than min_gain_split
        if best_gain < self.min_gain_split:
            return None, None, None

        split_idx = possible_split_positions[best_idx]
        threshold = (
            sorted_projections[split_idx - 1] + sorted_projections[split_idx]
        ) / 2.0

        # Return the absolute position of the split index
        return threshold, start + split_idx, best_gain

    def _build_tree(self, depth=0, start=0, end=None):
        if end is None:
            end = len(self.sorted_gradients)

        n_samples = end - start
        node = {"n_samples": n_samples}

        if depth == self.max_depth or n_samples < 2 * self.min_samples_leaf:
            node["is_leaf"] = True
            # Compute leaf value
            total_gradients = self.cumsum_gradients[end - 1] - (
                self.cumsum_gradients[start - 1] if start > 0 else 0
            )
            total_hessians = self.cumsum_hessians[end - 1] - (
                self.cumsum_hessians[start - 1] if start > 0 else 0
            )
            node["value"] = total_gradients / (total_hessians + 1e-10)
            return node

        threshold, split_idx, gain = self._find_best_split(start, end)

        if threshold is None:
            node["is_leaf"] = True
            # Compute leaf value
            total_gradients = self.cumsum_gradients[end - 1] - (
                self.cumsum_gradients[start - 1] if start > 0 else 0
            )
            total_hessians = self.cumsum_hessians[end - 1] - (
                self.cumsum_hessians[start - 1] if start > 0 else 0
            )
            node["value"] = total_gradients / (total_hessians + 1e-10)
            return node

        node.update(
            {
                "is_leaf": False,
                "threshold": threshold,
                "left": self._build_tree(depth + 1, start, split_idx),
                "right": self._build_tree(depth + 1, split_idx, end),
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
        if len(y) < 2 * self.min_samples_leaf:
            raise ValueError(
                f"Not enough samples for min_samples_leaf={self.min_samples_leaf}"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Compute probabilities and gradients
        logits = prev_scores
        probabilities = 1 / (1 + np.exp(-logits))

        gradients = y - probabilities
        hessians = probabilities * (1 - probabilities)

        self.direction = self._find_direction(X, y, prev_scores)
        projections = self._project_data(X)
        sort_idx = np.argsort(projections)

        # Cache sorted projections, gradients, and hessians
        self.sorted_projections = projections[sort_idx]
        self.sorted_gradients = gradients[sort_idx]
        self.sorted_hessians = hessians[sort_idx]

        # Precompute cumulative sums
        self.cumsum_gradients = np.cumsum(self.sorted_gradients)
        self.cumsum_hessians = np.cumsum(self.sorted_hessians)

        self.tree = self._build_tree(depth=0, start=0, end=len(y))
        return self

    def predict_proba(self, X, raw=False):
        X = np.asarray(X, dtype=np.float32)
        projections = self._project_data(X)
        predictions = np.zeros(len(X), dtype=np.float32)
        indices = np.arange(len(X))
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

        if raw:
            return predictions
        else:
            return 1 / (1 + np.exp(-predictions))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class SpaceBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=10,
        alpha=0.0,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain_split=0.0,
        bagging_fraction=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_split = min_gain_split
        self.bagging_fraction = bagging_fraction
        self.trees = []
        self.alpha = alpha
        self.init_prediction = None
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Initialize raw scores (logit of mean target)
        mean_y = np.mean(y)
        mean_y = np.clip(mean_y, 1e-6, 1 - 1e-6)  # Avoid division by zero
        self.init_prediction = np.log(mean_y / (1 - mean_y))
        current_predictions = np.full(
            len(y), self.init_prediction, dtype=np.float32
        )  # Raw scores

        rng = np.random.default_rng(self.random_state)  # Use numpy's random generator

        for i in range(self.n_estimators):
            # Generate bootstrap sample indices
            if self.bagging_fraction < 1.0:
                sample_size = int(len(y) * self.bagging_fraction)
                bootstrap_indices = rng.choice(len(y), size=sample_size, replace=True)
            else:
                bootstrap_indices = np.arange(len(y))

            # Create bootstrap samples
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            current_predictions_bootstrap = current_predictions[bootstrap_indices]

            # Train a tree on gradients
            tree = SpaceTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                alpha=self.alpha,
                min_gain_split=self.min_gain_split,
                random_state=(
                    self.random_state + i if self.random_state is not None else None
                ),
            )
            tree.fit(X_bootstrap, y_bootstrap, current_predictions_bootstrap)  # Pass raw scores
            self.trees.append(tree)

            # Update raw scores with tree predictions on all data
            current_predictions += self.learning_rate * tree.predict_proba(X, raw=True)

    def predict_proba(self, X, raw=False):
        X = np.asarray(X, dtype=np.float32)
        logits = np.full(len(X), self.init_prediction, dtype=np.float32)
        for tree in self.trees:
            logits += self.learning_rate * tree.predict_proba(X, raw=True)
        if raw:
            return logits
        else:
            return 1 / (1 + np.exp(-logits))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
