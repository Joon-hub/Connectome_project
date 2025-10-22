import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from typing import Tuple
import pickle

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
        print("\\nTraining model...")
        
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
        '''Perform subject-wise cross-validation'''
        print("\\nPerforming cross-validation...")
        
        cv_config = self.config.get('cross_validation')
        unique_subjects = np.unique(subjects)
        
        kf = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_subjects, np.zeros(len(unique_subjects))), 1):
            train_subjects = unique_subjects[train_idx]
            val_subjects = unique_subjects[val_idx]
            
            train_mask = np.isin(subjects, train_subjects)
            val_mask = np.isin(subjects, val_subjects)
            
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model = LogisticRegression(**self.config.get('model', 'params'))
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(acc)
            
            print(f"Fold {fold}: {acc:.4f}")
        
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