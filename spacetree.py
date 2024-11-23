import numpy as np

class QuantileNormDecisionTree:
    def __init__(self, max_depth=3, n_quantiles=4, min_samples_split=10, n_ref_points=5, 
                 impurity_method: str = 'mse'):
        self.max_depth = max_depth
        self.n_quantiles = n_quantiles
        self.min_samples_split = min_samples_split
        self.n_ref_points = n_ref_points
        self.impurity_method = impurity_method
        self.tree = None
        self.norm_matrix = None
        self.ref_points = None

    def _generate_reference_points(self, X):
        n_features = X.shape[1]
        ref_points = []

        origin = np.zeros(n_features)
        ref_points.append(origin)

        median_point = np.median(X, axis=0)
        ref_points.append(median_point)

        n_corners = min(n_features, 3)
        corners = np.array(list(np.ndindex((2,) * n_corners)))
        corners = corners * 2 - 1

        if n_features > n_corners:
            padding = np.zeros((len(corners), n_features - n_corners))
            corners = np.hstack([corners, padding])

        X_std = np.std(X, axis=0)
        X_mean = np.mean(X, axis=0)
        corners = corners * X_std + X_mean
        ref_points.extend(corners)

        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        n_random = max(0, self.n_ref_points - len(ref_points))
        if n_random > 0:
            random_points = np.random.uniform(
                X_min, X_max, 
                size=(n_random, n_features)
            )
            ref_points.extend(random_points)
        
        return np.array(ref_points)

    def _compute_norm_matrix(self, X):
        """Precompute norms for all points to all reference points."""
        n_samples = X.shape[0]
        n_ref_points = self.ref_points.shape[0]
        norm_matrix = np.zeros((n_samples, n_ref_points))
        
        for i in range(n_ref_points):
            norm_matrix[:, i] = np.linalg.norm(X - self.ref_points[i], axis=1)
        
        return norm_matrix

    def _find_best_split(self, indices, y, depth):
        if len(indices) < self.min_samples_split:
            return None, None, None, float('inf')
            
        best_impurity = float('inf')
        best_threshold = None
        best_ref_idx = None
        best_split_indices = None
        
        for ref_idx in range(self.ref_points.shape[0]):
            norms = self.norm_matrix[indices, ref_idx]
            
            quantiles = np.percentile(
                norms,
                np.linspace(0, 100, self.n_quantiles + 1)[1:-1]
            )
            
            for threshold in quantiles:
                left_mask = norms <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_split or 
                    np.sum(right_mask) < self.min_samples_split):
                    continue
                
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]
                
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                n_left = len(left_indices)
                n_right = len(right_indices)
                n_total = n_left + n_right
                
                total_impurity = (n_left / n_total) * left_impurity + \
                                (n_right / n_total) * right_impurity
                
                if total_impurity < best_impurity:
                    best_impurity = total_impurity
                    best_threshold = threshold
                    best_ref_idx = ref_idx
                    best_split_indices = (left_indices, right_indices)
        
        return best_threshold, best_split_indices, best_ref_idx, best_impurity

    def _calculate_impurity(self, y):
        if len(y) == 0:
            return 0
            
        y_mean = np.mean(y)
        n_samples = len(y)
        
        if self.impurity_method == 'mse':
            return np.mean((y - y_mean) ** 2)
        elif self.impurity_method == 'se':
            return np.sum((y - y_mean) ** 2)
        elif self.impurity_method == 'rmse':
            return np.sqrt(np.mean((y - y_mean) ** 2))
        else:
            raise ValueError(f"Unknown impurity method: {self.impurity_method}")

    def _build_tree(self, indices, y, depth=0):
        if (depth == self.max_depth or 
            len(indices) <= self.min_samples_split or 
            len(np.unique(y[indices])) == 1):
            return {
                'value': np.mean(y[indices]),
                'is_leaf': True,
                'n_samples': len(indices),
                'impurity': self._calculate_impurity(y[indices])
            }

        best_threshold, best_split_indices, best_ref_idx, best_impurity = \
            self._find_best_split(indices, y, depth)
        
        if best_threshold is None:
            return {
                'value': np.mean(y[indices]),
                'is_leaf': True,
                'n_samples': len(indices),
                'impurity': self._calculate_impurity(y[indices])
            }

        left_indices, right_indices = best_split_indices

        return {
            'threshold': best_threshold,
            'ref_idx': best_ref_idx,
            'left': self._build_tree(left_indices, y, depth + 1),
            'right': self._build_tree(right_indices, y, depth + 1),
            'is_leaf': False,
            'n_samples': len(indices),
            'impurity': best_impurity
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.ref_points = self._generate_reference_points(X)
        self.norm_matrix = self._compute_norm_matrix(X)
        indices = np.arange(len(X))
        self.tree = self._build_tree(indices, y)
        return self

    def predict(self, X):
        X = np.array(X)
        
        def _predict_single(tree, x):
            if tree['is_leaf']:
                return tree['value']
            
            x_norm = np.linalg.norm(x - self.ref_points[tree['ref_idx']])
            
            if x_norm <= tree['threshold']:
                return _predict_single(tree['left'], x)
            else:
                return _predict_single(tree['right'], x)

        return np.array([_predict_single(self.tree, x) for x in X])
