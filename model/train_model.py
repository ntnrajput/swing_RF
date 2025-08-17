# model/train_model.py

import sys
import os
import joblib
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from time import sleep
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.feature_selection import SelectFromModel



import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger
from config import FEATURE_COLUMNS, MODEL_FILE

logger = get_logger(__name__)

def train_model(df_all, available_features=None, model_params=None):
    """
    Enhanced trading model with hyperparameter tuning and volume analysis
    
    Args:
        df_all: DataFrame with all features and target
        available_features: List of feature column names (optional, can be imported from config)
        model_params: Optional hyperparameter grid for tuning
    
    Returns:
        tuple: (best_model, scaler, available_features)
    """
    # Import FEATURE_COLUMNS if not provided
    if available_features is None:
        try:
            from config import FEATURE_COLUMNS
            available_features = FEATURE_COLUMNS
        except ImportError:
            raise ValueError("available_features must be provided or FEATURE_COLUMNS must be available in config")
    
    # Validate that all features exist in the dataframe
    missing_features = [col for col in available_features if col not in df_all.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    # Also check if target column exists
    if 'target_hit' not in df_all.columns:
        raise ValueError("Target column 'target_hit' not found in dataframe")
    try:
        print("📊 Initial data shape:", df_all.shape)
        initial_rows = len(df_all)
        # Replace inf/-inf in numeric columns only
        num_cols = df_all.select_dtypes(include=[np.number]).columns
        df_all[num_cols] = df_all[num_cols].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN in features or target
        df_all = df_all.dropna(subset=available_features + ['target_hit']).reset_index(drop=True)

        final_rows = len(df_all)
        print(f"✅ Dropped {initial_rows - final_rows} rows due to NaN/Inf values.")
        print("📊 Final data shape:", df_all.shape)

        # df_all.to_csv('df_all_with_features_for_traingin.csv')

        # Feature-target split
        X = df_all[available_features]
        y = df_all['target_hit']

        
        # Check class distribution
        class_counts = y.value_counts()
        class_percent = y.value_counts(normalize=True) * 100
        logger.info(f"Class distribution: {class_counts.to_dict()} ({class_percent.to_dict()} %)")


        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print("After balancing classes:", X.shape, "Class distribution:", y.value_counts())
        
        if class_percent.min() < 5:
            logger.warning("⚠ Severe class imbalance detected — consider resampling.")
        
        if len(class_counts) < 2:
            logger.error("Only one class found in target variable")
            raise ValueError("Only one class found in target variable")
            
        # Check if we have enough samples after cleaning
        MIN_SAMPLES = 500  # adjustable threshold
        if len(df_all) < MIN_SAMPLES:
            logger.warning(f"⚠ Very few samples remaining after cleaning: {len(df_all)} (< {MIN_SAMPLES})")
            
        # Additional check for target variable data types
        if not pd.api.types.is_numeric_dtype(y):
            logger.warning(f"Target variable dtype is {y.dtype}, converting to numeric")
            y = pd.to_numeric(y, errors='coerce')
            mask = ~y.isnull()
            X, y = X[mask], y[mask]
            logger.info(f"After numeric conversion: {len(X)} samples remain")
                
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ------------------------------------

        rf_for_fs = RandomForestClassifier(n_estimators=200, random_state=42)
        selector = SelectFromModel(rf_for_fs, threshold=0.005)  # Keep features above median importance
        selector.fit(X_scaled, y)
        X_selected = selector.transform(X_scaled)
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]
        print(f" Selected Features: {selected_features}")

        # Get names of selected features
        selected_mask = selector.get_support()
        selected_features = [f for f, keep in zip(available_features, selected_mask) if keep]
        print(f" Selected Top {len(selected_features)} Features: {selected_features}")
        
        
        # Split data with stratification (replaces time-aware split)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, stratify=y, test_size=0.2, random_state=42
        )

        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Enhanced hyperparameter tuning
        if model_params is None:
            model_params = {
                'n_estimators': [100],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [3],
                'max_features': ['sqrt']
            }
        
        print("🔍 Performing enhanced hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, model_params, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        print("\n Enhanced Model Evaluation:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Separate volume features importance
        volume_features = feature_importance[feature_importance['feature'].str.contains('volume|vwap|obv|ad_line')]
        print(f"\n Volume Features Importance (Top 10):")
        print(volume_features.head(10))
        
        # Plot feature importance and save instead of showing
        try:
            plt.figure(figsize=(12, 10))                        
            sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
            plt.title('Enhanced Feature Importance with Volume Analysis')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print("📊 Feature importance plot saved as 'feature_importance_plot.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")
        
        # Calculate additional metrics for comparison with original
        train_pred = best_model.predict(X_train)
        train_accuracy = np.mean(train_pred == y_train)
        test_accuracy = np.mean(y_pred == y_test)
        overfitting_gap = train_accuracy - test_accuracy
        win_rate = test_accuracy * 100
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overfitting_gap': overfitting_gap,
            'win_rate': win_rate,
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"Model metrics: {metrics}")
        
        # Create enhanced model pipeline
        model_pipeline = {
            'scaler': scaler,
            'selector': selector,
            'model': best_model,
            'feature_columns': selected_features,
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'feature_importance': feature_importance
        }
        
        # Save model and scaler
        joblib.dump(best_model, "enhanced_swing_model_with_volume.pkl")
        full_pipeline = Pipeline([
            ('scaler', scaler),
            ('selector', selector),
            ('model', best_model)
        ])

        # Bundle full model with metadata
        model_bundle = {
                'pipeline': full_pipeline,
                'feature_columns': selected_features,     # Features after selection
                'all_features': available_features,       # Features before selection (needed for backtest)
                'metrics': metrics,
                'best_params': grid_search.best_params_,
                'feature_importance': feature_importance
            }
        # Save single model bundle
        joblib.dump(model_bundle, MODEL_FILE)
        logger.info(f" Model pipeline saved to {MODEL_FILE}")
        




        print("Enhanced model with volume analysis saved!")
        
        # Trading recommendations
        logger.info("\nTRADING RECOMMENDATIONS:")
        if overfitting_gap < 0.05:
            logger.info("  Model shows good generalization")
        else:
            logger.warning("  Model may still be overfitting - use with caution")
        
        if win_rate >= 55:
            logger.info("  Model shows reasonable predictive power")
        else:
            logger.info("  Model needs improvement - consider more data or different features")
        
        logger.info(f"  Enhanced approach with ROC AUC: {metrics['roc_auc']:.4f} (win rate: {win_rate:.1f}%)")
        
        return model_bundle
        
    except Exception as e:
        logger.error(f"Failed to train enhanced model: {e}", exc_info=True)
        raise

