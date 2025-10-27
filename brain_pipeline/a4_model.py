import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Tuple
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class BrainRegionClassifier:
    '''Train and apply brain region classifier'''
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self._initialize_model()
    
    def _initialize_model(self):
        '''Initialize logistic regression model'''
        params = self.config.get('model', 'params')
        self.model = LogisticRegression(**params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        '''Train model on full dataset'''
        print("\nTraining model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Training accuracy: {accuracy:.4f}")
        return accuracy
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> dict:
        print("\nPerforming cross-validation (group-wise by subject)...")

        # Create the group sampler: each unique subject is a group
        grp = GroupKFold(n_splits=self.config.get('cross_validation', 'n_splits'))

        fold_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(grp.split(X, y, groups=subjects), start=1):
            print(f"\nFold {fold} —")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # scale and train model
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = LogisticRegression(**self.config.get('model', 'params'))
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            acc = accuracy_score(y_val, y_pred)
            print(f"  Accuracy: {acc:.4f}")
            fold_accuracies.append(acc)

        return {
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies)
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Predict on new data'''
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        return y_pred, y_proba
    
    def save(self, filepath: str):
        '''Save model and scaler'''
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        '''Load model and scaler'''
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
        print(f"Model loaded from {filepath}")