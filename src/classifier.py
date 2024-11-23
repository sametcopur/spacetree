import numpy as np
from sklearn.linear_model import Ridge
from scipy.special import expit


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

    def _find_direction(self, X, y):
        lr = Ridge(alpha=self.alpha, random_state=self.random_state)
        lr.fit(X, y)
        direction = lr.coef_
        return direction / np.sqrt(np.sum(direction**2))

    def _project_data(self, X):
        return X @ self.direction
    
    def compute_entropy(self, probs):
        epsilon = 1e-10  # Numerik kararlılık için küçük bir değer
        probs = np.clip(probs, epsilon, 1 - epsilon)  # Olasılıkları [epsilon, 1-epsilon] aralığında sıkıştır
        return -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)


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

        left_counts = np.cumsum(sorted_y)[split_indices - 1]
        total_counts = np.sum(sorted_y)
        right_counts = total_counts - left_counts

        left_n = split_indices
        right_n = n_samples - split_indices

        valid_splits = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
        if not np.any(valid_splits):
            return None, None, None

        left_probs = left_counts / left_n
        right_probs = right_counts / right_n

        left_entropy =  self.compute_entropy(left_probs)
        right_entropy = self.compute_entropy(right_probs)

        weighted_entropy = (left_n * left_entropy + right_n * right_entropy) / n_samples
        best_idx = np.argmin(weighted_entropy[valid_splits])
        best_threshold = sorted_projections[split_indices[valid_splits][best_idx]]
        best_split_mask = projections <= best_threshold

        return best_threshold, best_split_mask, weighted_entropy[valid_splits][best_idx]

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        node = {"n_samples": n_samples, "value": np.mean(y)}

        if depth == self.max_depth or n_samples < 2 * self.min_samples_split:
            node["is_leaf"] = True
            node["value"] = np.mean(y)  # Leaf node value is the mean probability
            return node

        threshold, split_mask, score = self._find_best_split(X, y)

        if threshold is None:
            node["is_leaf"] = True
            node["value"] = np.mean(y)
            return node

        X_left, y_left = X[split_mask], y[split_mask]
        X_right, y_right = X[~split_mask], y[~split_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            node["is_leaf"] = True
            node["value"] = np.mean(y)
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

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        probabilities = np.zeros(len(X), dtype=np.float32)
        nodes = [self.tree] * len(X)
        mask = np.ones(len(X), dtype=bool)

        while np.any(mask):
            leaf_mask = mask & np.array([node["is_leaf"] for node in nodes])
            if np.any(leaf_mask):
                probabilities[leaf_mask] = np.array(
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

        return probabilities

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

class SpaceBoostingClassifier:
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
        self.init_log_odds = None
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Başlangıç Log-Odds
        positive_rate = np.clip(np.mean(y), 1e-15, 1 - 1e-15)  # Olasılığın 0 veya 1 olmamasını sağla
        self.init_log_odds = np.log(positive_rate / (1 - positive_rate))  # Başlangıç log-odds
        current_predictions = np.full(len(y), self.init_log_odds, dtype=np.float32)  # Başlangıç tahminleri

        for i in range(self.n_estimators):
            # Residual: raw score ile hedef değer arasındaki fark
            residuals = y - current_predictions  # Residual, raw score üzerinden hesaplanır
            
            # Ağacı fit et
            tree = SpaceTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_splits=self.n_splits,
                alpha=self.alpha,
                random_state=(self.random_state + i if self.random_state is not None else None),
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Raw skorları güncelle
            current_predictions += self.learning_rate * tree.predict(X)  # Her ağacın katkısını ekle

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        predictions = np.full(len(X), self.init_log_odds, dtype=np.float32)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        # Sigmoid ile olasılık tahmini
        probabilities = expit(predictions)  # Raw skorları sigmoid ile olasılığa çevir
        return np.vstack([1 - probabilities, probabilities]).T  # [P(0), P(1)]

    def predict(self, X):
        # Sınıf tahmini
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
