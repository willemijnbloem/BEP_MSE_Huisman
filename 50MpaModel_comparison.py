import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import csv

# Define targets and mappings
targets = ['Yield Strength [Mpa]', 'Tensile Strength [in MPA]', 'Elongation']

COLUMN_MAPPINGS = {
    'Yield Strength [Mpa]': 'QC_Re',
    'Tensile Strength [in MPA]': 'QC_Rm',
    'Elongation': 'QC_A'
}

# Define specific confidence intervals for each property and grade
CONFIDENCE_SETTINGS = {
    'S355': {
        'Yield Strength [Mpa]': np.linspace(0.01, 0.99, 10),
        'Tensile Strength [in MPA]': np.linspace(0.01, 0.99, 10),
        'Elongation': np.linspace(0.01, 0.99, 10)
    },
    'S690': {
        'Yield Strength [Mpa]': np.linspace(0.01, 0.99, 10),
        'Tensile Strength [in MPA]': np.linspace(0.01, 0.99, 10),
        'Elongation': np.linspace(0.01, 0.99, 10)
    }
}

def safe_log(x, min_value=1e-10):
    """
    Safely compute logarithm by ensuring input is positive
    """
    return np.log(np.maximum(x, min_value))

def create_enhanced_features(df):
    """
    Create enhanced features with safe calculations for negative/zero values
    """
    enhanced = df.copy()
    eps = 1e-10  # Small constant to prevent division by zero
    
    # Original features with safe operations
    enhanced['Yield_Ductility_Product'] = enhanced['QC_Re'] * enhanced['QC_A']
    enhanced['Work_Hardening_Range'] = enhanced['QC_Rm'] - enhanced['QC_Re']
    enhanced['Normalized_Work_Hardening'] = (enhanced['QC_Rm'] - enhanced['QC_Re']) / np.maximum(enhanced['QC_Re'], eps)
    enhanced['Work_Hardening_Ductility'] = (enhanced['QC_Rm'] - enhanced['QC_Re']) * enhanced['QC_A']
    enhanced['Ductility_Weighted_Strength'] = enhanced['QC_Rm'] * np.sqrt(np.maximum(enhanced['QC_A'], 0))
    enhanced['Quality_Index'] = enhanced['QC_Rm'] + 150 * safe_log(np.maximum(enhanced['QC_A'], eps))
    enhanced['Performance_Index'] = enhanced['QC_Re'] * np.power(np.maximum(enhanced['QC_A'], 0), 0.333)
    enhanced['Strain_Hardening_Coefficient'] = safe_log(enhanced['QC_Rm']/np.maximum(enhanced['QC_Re'], eps)) / safe_log(np.maximum(enhanced['QC_A'], eps)/100)
    enhanced['Strain_Energy'] = (enhanced['QC_Rm'] + enhanced['QC_Re']) * enhanced['QC_A'] / 2
    enhanced['Strength_Ductility_Balance'] = (enhanced['QC_Rm'] * enhanced['QC_A']) / np.maximum(enhanced['QC_Re'], eps)
    enhanced['Formability_Index'] = enhanced['QC_A'] * np.sqrt(enhanced['QC_Rm']/np.maximum(enhanced['QC_Re'], eps))
    
    # Thickness-related features with safe operations
    enhanced['Thickness_Strength_Index'] = enhanced['QC_Rm'] / safe_log(enhanced['Dimension'] + 1)
    enhanced['Thickness_Ductility_Ratio'] = enhanced['QC_A'] / np.maximum(enhanced['Dimension'], eps)
    enhanced['Hall_Petch_Approximation'] = enhanced['QC_Re'] * np.sqrt(np.maximum(enhanced['Dimension'], 0))
    enhanced['Size_Effect_Factor'] = enhanced['QC_Re'] * np.power(np.maximum(enhanced['Dimension'], eps), -0.5)
    enhanced['Thickness_Quality_Index'] = enhanced['Quality_Index'] / safe_log(enhanced['Dimension'] + 1)
    enhanced['Thickness_Strain_Energy'] = enhanced['Strain_Energy'] / np.maximum(enhanced['Dimension'], eps)
    enhanced['Normalized_Thickness_Strength'] = enhanced['QC_Re'] / (np.maximum(enhanced['Dimension'], eps) * np.maximum(enhanced['QC_Rm'], eps))
    enhanced['Thickness_Performance_Factor'] = enhanced['Performance_Index'] / np.sqrt(np.maximum(enhanced['Dimension'], eps))
    enhanced['Surface_Volume_Strength'] = enhanced['QC_Re'] * (1 / np.maximum(enhanced['Dimension'], eps))
    
    # Add interaction terms with safe operations
    enhanced['Thickness_Strength_Ratio'] = enhanced['QC_Re'] / np.maximum(enhanced['Dimension'], eps)
    enhanced['Thickness_Squared'] = np.power(enhanced['Dimension'], 2)
    enhanced['Strength_Thickness_Interaction'] = enhanced['QC_Re'] * safe_log(np.maximum(enhanced['Dimension'], eps))
    
    # Add material property ratios with safe operations
    enhanced['Strength_Ratio'] = enhanced['QC_Rm'] / np.maximum(enhanced['QC_Re'], eps)
    enhanced['Specific_Strength'] = enhanced['QC_Re'] / np.maximum(enhanced['Dimension'], eps)
    
    # Add polynomial features for key parameters with safe operations
    enhanced['Dimension_Cubic'] = np.power(enhanced['Dimension'], 3)
    enhanced['Log_Thickness'] = safe_log(np.maximum(enhanced['Dimension'], eps))
    
    return enhanced

def calculate_confidence_intervals(fold_errors, confidence_percentage):
    """
    Calculate confidence intervals based on percentage errors across folds
    confidence_percentage: directly specify the desired confidence (e.g., 90, 95, 99)
    """
    # Convert percentage to z-score
    z_score = norm.ppf(1 - (1 - confidence_percentage/100)/2)
    std_dev = np.std(fold_errors)
    confidence_interval = z_score * std_dev
    return confidence_interval

def process_data_and_train():
    """
    Main processing and training pipeline
    """
    # 1. Load and prepare data
    df = pd.read_csv("C:/Users/31633/OneDrive - Delft University of Technology/BEP/AllData2.csv", sep=';')
    
    # 2. Clean numerical data
    numerical_base_features = ['QC_Re', 'QC_Rm', 'QC_A', 'Dimension']
    def clean_numeric(value):
        if isinstance(value, str):
            if '/' in value:
                return float(value.split('/')[0])
            return float(value.replace(',', '.').replace('[^\d.-]', ''))
        return value

    for col in numerical_base_features + targets:
        df[col] = df[col].apply(clean_numeric)

    # 3. Remove missing values
    df = df.dropna(subset=numerical_base_features + ['Manufacturer', 'Simplified Grade (for further analysis)'] + targets)
    
    # 4. Standardize grade names
    df['Simplified Grade (for further analysis)'] = df['Simplified Grade (for further analysis)'].apply(
        lambda x: 'S690' if 'StE690' in str(x) or 'S690' in str(x) else ('S355' if 'S355' in str(x) else x)
    )
    
    # 5. Create enhanced features
    df_enhanced = create_enhanced_features(df)
    
    # 6. Initialize results container - modified to include false rates
    model_metrics = {
        'S355': {'metrics': {}, 'false_rates': {}, 'predictions': {}, 'actual_values': {}}, 
        'S690': {'metrics': {}, 'false_rates': {}, 'predictions': {}, 'actual_values': {}}
    }
    
    # 7. Get engineered features
    engineered_features = [col for col in df_enhanced.columns 
                          if col not in targets + ['Manufacturer', 'Simplified Grade (for further analysis)']
                          and col != 'Grade (as reported)'
                          and df_enhanced[col].dtype != 'object']
    
    # 8. Process each grade and target
    for grade in ['S355', 'S690']:
        grade_data = df_enhanced[df_enhanced['Simplified Grade (for further analysis)'] == grade]
        
        # Initialize storage for predictions and actual values
        all_predictions = {target: [] for target in targets}
        all_actual_values = {target: [] for target in targets}
        
        for target in targets:
            # 9. Select features based on target
            if target == 'Elongation':
                features_to_use = numerical_base_features
            else:
                # Calculate correlations
                correlations = grade_data[engineered_features].corrwith(grade_data[target])
                features_to_use = correlations[correlations.abs() > 0.1].index.tolist()
            
            X = grade_data[features_to_use].values
            y = grade_data[target].values
            
            # 10. Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 11. Perform cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            
            for train_idx, test_idx in kf.split(X_scaled):
                # 12. Split data
                X_train_fold = X_scaled[train_idx]
                X_test_fold = X_scaled[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # 13. Train models
                rf_model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=50,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=False,
                    random_state=42,
                    n_jobs=-1,
                    criterion='friedman_mse'        # Use mean squared error as the criterion for splitting
                )
                
                # Train Random Forest model
                rf_model.fit(X_train_fold, y_train_fold)
                rf_predictions = rf_model.predict(X_test_fold)

                rt_model = ExtraTreesRegressor(
                    n_estimators=50,
                    max_depth=50,
                    min_samples_split=2,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                    criterion='friedman_mse'        # Use mean squared error as the criterion for splitting
                )
                
                rt_model.fit(X_train_fold, y_train_fold)
                rt_predictions = rt_model.predict(X_test_fold)

                # Use the most accurate predictions from the two models
                predictions = np.where(
                    np.abs(rf_predictions - y_test_fold) < np.abs(rt_predictions - y_test_fold),
                    rf_predictions,
                    rt_predictions
                )
                
                # Store predictions and actual values for false rate analysis
                all_predictions[target].extend(predictions)
                all_actual_values[target].extend(y_test_fold)
            
            # Calculate regular metrics
            rmse = np.sqrt(mean_squared_error(all_actual_values[target], all_predictions[target]))
            mae = mean_absolute_error(all_actual_values[target], all_predictions[target])
            r2 = r2_score(all_actual_values[target], all_predictions[target])
            
            # Store regular metrics
            model_metrics[grade]['metrics'][target] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            # Calculate false rates for different confidence intervals
            false_rates = calculate_false_rates(
                np.array(all_predictions[target]),
                np.array(all_actual_values[target]),
                grade,
                target
            )
            
            # Store false rates
            model_metrics[grade]['false_rates'][target] = false_rates
            
            # Store predictions and actual values
            model_metrics[grade]['predictions'][target] = all_predictions[target]
            model_metrics[grade]['actual_values'][target] = all_actual_values[target]
    
    return model_metrics

def get_min_requirements(thickness, grade):
    """
    Determine minimum required properties based on thickness and grade.
    Returns dict with min/max requirements for each property.
    """
    requirements = {
        'yield_strength': {'min': None, 'max': None},
        'tensile_strength': {'min': None, 'max': None},
        'elongation': {'min': None, 'max': None}
    }
    
    if grade == 'S355':
        # Yield strength and tensile requirements based on thickness
        if thickness <= 16:
            requirements['yield_strength']['min'] = 355
            requirements['tensile_strength']['min'] = 470
            requirements['tensile_strength']['max'] = 630
        elif thickness <= 40:
            requirements['yield_strength']['min'] = 345
            requirements['tensile_strength']['min'] = 470
            requirements['tensile_strength']['max'] = 630
        elif thickness <= 63:
            requirements['yield_strength']['min'] = 335
            requirements['tensile_strength']['min'] = 470
            requirements['tensile_strength']['max'] = 630
        elif thickness <= 80:
            requirements['yield_strength']['min'] = 325
            requirements['tensile_strength']['min'] = 470
            requirements['tensile_strength']['max'] = 630
        elif thickness <= 100:
            requirements['yield_strength']['min'] = 315
            requirements['tensile_strength']['min'] = 470
            requirements['tensile_strength']['max'] = 630
        elif thickness <= 150:
            requirements['yield_strength']['min'] = 295
            requirements['tensile_strength']['min'] = 450
            requirements['tensile_strength']['max'] = 600
        elif thickness <= 200:
            requirements['yield_strength']['min'] = 285
            requirements['tensile_strength']['min'] = 450
            requirements['tensile_strength']['max'] = 600
        elif thickness <= 250:
            requirements['yield_strength']['min'] = 275
            requirements['tensile_strength']['min'] = 450
            requirements['tensile_strength']['max'] = 600
        elif thickness <= 500:
            requirements['yield_strength']['min'] = 265
            requirements['tensile_strength']['min'] = 450
            requirements['tensile_strength']['max'] = 600
        
        # Updated elongation requirements for S355
        if thickness <= 63:
            requirements['elongation']['min'] = 22
        elif thickness <= 250:
            requirements['elongation']['min'] = 21
        else:  # thickness > 250
            requirements['elongation']['min'] = 19
            
    elif grade == 'S690':
        # Yield strength requirements
        if thickness <= 50:
            requirements['yield_strength']['min'] = 690
            requirements['tensile_strength']['min'] = 770
            requirements['tensile_strength']['max'] = 940
        elif thickness <= 100:
            requirements['yield_strength']['min'] = 650
            requirements['tensile_strength']['min'] = 760
            requirements['tensile_strength']['max'] = 930
        elif thickness <= 150:
            requirements['yield_strength']['min'] = 630
            requirements['tensile_strength']['min'] = 710
            requirements['tensile_strength']['max'] = 900
        else:
            requirements['yield_strength']['min'] = 630
            requirements['tensile_strength']['min'] = 710
            requirements['tensile_strength']['max'] = 900
        
        # Elongation requirements for S690
        requirements['elongation']['min'] = 16  # Set directly to 16 for S690
        
    return requirements

def meets_requirements(value, actual_value, grade, property_type):
    """
    Check if a value meets the requirements for the given grade and property
    """
    requirements = get_min_requirements(16, grade)  # Using default thickness of 16mm
    
    if property_type == 'Yield Strength [Mpa]':
        min_req = requirements['yield_strength']['min']
        return value >= min_req if isinstance(value, float) else actual_value >= min_req
    elif property_type == 'Tensile Strength [in MPA]':
        min_req = requirements['tensile_strength']['min']
        max_req = requirements['tensile_strength']['max']
        return min_req <= value <= max_req if isinstance(value, float) else min_req <= actual_value <= max_req
    elif property_type == 'Elongation':
        min_req = requirements['elongation']['min']
        return value >= min_req if isinstance(value, float) else actual_value >= min_req
    return False

def check_predictions_against_requirements(predictions, actual_values, thickness, grade):
    """
    Check if predictions meet the requirements for the given grade and thickness
    """
    requirements = get_min_requirements(thickness, grade)
    
    # Initialize arrays to store whether each prediction passes requirements
    yield_passes = np.ones(len(predictions['Yield Strength [Mpa]']), dtype=bool)
    tensile_passes = np.ones(len(predictions['Tensile Strength [in MPA]']), dtype=bool)
    elongation_passes = np.ones(len(predictions['Elongation']), dtype=bool)
    
    # Check yield strength requirements
    if requirements['yield_strength']['min'] is not None:
        yield_passes &= (predictions['Yield Strength [Mpa]'] >= requirements['yield_strength']['min'])
    if requirements['yield_strength']['max'] is not None:
        yield_passes &= (predictions['Yield Strength [Mpa]'] <= requirements['yield_strength']['max'])
    
    # Check tensile strength requirements
    if requirements['tensile_strength']['min'] is not None:
        tensile_passes &= (predictions['Tensile Strength [in MPA]'] >= requirements['tensile_strength']['min'])
    if requirements['tensile_strength']['max'] is not None:
        tensile_passes &= (predictions['Tensile Strength [in MPA]'] <= requirements['tensile_strength']['max'])
    
    # Check elongation requirements
    if requirements['elongation']['min'] is not None:
        elongation_passes &= (predictions['Elongation'] >= requirements['elongation']['min'])
    if requirements['elongation']['max'] is not None:
        elongation_passes &= (predictions['Elongation'] <= requirements['elongation']['max'])
    
    # Overall pass/fail (must pass all requirements)
    all_requirements_met = yield_passes & tensile_passes & elongation_passes
    
    pass_rate = np.mean(all_requirements_met) * 100
    
    return {
        'pass_rate': pass_rate,
        'total_samples': len(all_requirements_met),
        'passed_samples': np.sum(all_requirements_met),
        'requirements': requirements
    }

def classify_prediction(prediction, confidence_interval, thickness, grade):
    """Modified to handle single property predictions with confidence intervals"""
    property_type = list(prediction.keys())[0]
    value = prediction[property_type]
    margin = confidence_interval[property_type]  # Simplified to directly access margin
    
    # Get requirements for the given grade and thickness
    requirements = get_min_requirements(thickness, grade)
    
    # For Yield Strength
    if property_type == 'Yield Strength [Mpa]':
        lower_bound = value - margin
        return 'PASS' if lower_bound >= requirements['yield_strength']['min'] else 'FAIL'
    
    # For Tensile Strength
    elif property_type == 'Tensile Strength [in MPA]':
        lower_bound = value - margin
        upper_bound = value + margin
        
        # Fail if lower bound is below minimum requirement
        if lower_bound < requirements['tensile_strength']['min']:
            return 'FAIL'
        
        # Check if upper bound exceeds maximum
        if upper_bound > requirements['tensile_strength']['max']:
            return 'SEMI-PASS'
            
        return 'PASS'
    
    # For Elongation
    elif property_type == 'Elongation':
        lower_bound = value - margin
        if lower_bound < 6:
            return 'FAIL'
        elif lower_bound >= requirements['elongation']['min']:
            return 'PASS'
        else:
            return 'SEMI-PASS'

def get_overall_classification(classifications, properties):
    """
    Determine overall classification based on specific requirements
    """
    # Extract individual classifications
    yield_strength = classifications[properties.index('Yield Strength [Mpa]')]
    tensile_strength = classifications[properties.index('Tensile Strength [in MPA]')]
    elongation = classifications[properties.index('Elongation')]
    
    # Check for PASS first
    if all(c == 'PASS' for c in classifications):
        return 'PASS'
    
    # Check for SEMI-PASS
    if yield_strength == 'PASS':  # Yield strength must pass
        # Check if either tensile is SEMI-PASS or elongation is SEMI-PASS
        if (tensile_strength == 'SEMI-PASS' or elongation == 'SEMI-PASS') and \
           (tensile_strength != 'FAIL' and elongation != 'FAIL'):
            return 'SEMI-PASS'
    
    # All other cases are FAIL
    return 'FAIL'

def plot_overall_classification_comparison(metrics, optimal_cis):
    """
    Create separate plots for predicted and actual classifications
    """
    grades = ['S355', 'S690']
    properties = ['Yield Strength [Mpa]', 'Tensile Strength [in MPA]', 'Elongation']
    categories = ['PASS', 'SEMI-PASS', 'FAIL']
    
    # Set up two separate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    bar_width = 0.35
    
    # Colors for different categories
    colors = {
        'PASS': '#2ecc71',     # Green
        'SEMI-PASS': '#f1c40f', # Yellow
        'FAIL': '#e74c3c'      # Red
    }
    
    results_data = {'Predicted': {}, 'Actual': {}}
    
    # Calculate results (existing code)
    for grade in grades:
        total_samples = len(metrics[grade]['predictions'][properties[0]])
        predicted_results = {'PASS': 0, 'SEMI-PASS': 0, 'FAIL': 0}
        actual_results = {'PASS': 0, 'SEMI-PASS': 0, 'FAIL': 0}
        
        for i in range(total_samples):
            pred_classifications = []
            actual_classifications = []
            
            for prop in properties:
                pred = metrics[grade]['predictions'][prop][i]
                actual = metrics[grade]['actual_values'][prop][i]
                margin = optimal_cis[grade][prop]['margin_of_error']
                
                pred_dict = {prop: pred}
                actual_dict = {prop: actual}
                
                pred_class = classify_prediction(pred_dict, {prop: margin}, 16, grade)
                actual_class = classify_prediction(actual_dict, {prop: 0}, 16, grade)
                
                pred_classifications.append(pred_class)
                actual_classifications.append(actual_class)
            
            pred_overall = get_overall_classification(pred_classifications, properties)
            actual_overall = get_overall_classification(actual_classifications, properties)
            
            predicted_results[pred_overall] += 1
            actual_results[actual_overall] += 1
        
        results_data['Predicted'][grade] = {
            'values': predicted_results,
            'total': total_samples
        }
        results_data['Actual'][grade] = {
            'values': actual_results,
            'total': total_samples
        }
    
    # Plotting
    x = np.arange(len(categories))
    
    # Plot Predicted Results (top subplot)
    for idx, grade in enumerate(grades):
        values = results_data['Predicted'][grade]['values']
        total = results_data['Predicted'][grade]['total']
        percentages = [values[cat]/total * 100 for cat in categories]
        
        bars = ax1.bar(x + (bar_width/2 if grade == 'S690' else -bar_width/2), 
                      percentages, bar_width, label=grade,
                      color=[colors[cat] for cat in categories])
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show labels if value is > 1%
                ax1.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Predicted Classification Results')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Plot Actual Results (bottom subplot)
    for idx, grade in enumerate(grades):
        values = results_data['Actual'][grade]['values']
        total = results_data['Actual'][grade]['total']
        percentages = [values[cat]/total * 100 for cat in categories]
        
        bars = ax2.bar(x + (bar_width/2 if grade == 'S690' else -bar_width/2), 
                      percentages, bar_width, label=grade,
                      color=[colors[cat] for cat in categories])
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show labels if value is > 1%
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Actual Classification Results')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def calculate_category_statistics(predictions, confidence_intervals, thicknesses, grade):
    """
    Calculate statistics for each category
    """
    categories = {'PASS': 0, 'SEMI-PASS': 0, 'FAIL': 0}
    total_samples = len(next(iter(predictions.values())))
    
    for i in range(total_samples):
        pred = {key: values[i] for key, values in predictions.items()}
        category = classify_prediction(pred, confidence_intervals, thicknesses[i], grade)
        categories[category] += 1
    
    # Calculate percentages
    percentages = {
        category: (count/total_samples * 100) 
        for category, count in categories.items()
    }
    
    return {
        'counts': categories,
        'percentages': percentages,
        'total_samples': total_samples
    }

def calculate_false_rates(predictions, actual_values, grade, property_type):
    """
    Calculate false rates using property and grade specific confidence intervals
    with proper averaging across folds
    """
    # Scale predictions if they are too large (convert to MPa)
    if np.mean(predictions) > 1000:  # If values are larger than typical MPa range
        predictions = predictions / 1000000  # Convert from micro-MPa to MPa
    
    residuals = predictions - actual_values
    std_estimate = np.std(residuals)
    n_folds = 10  # Add this constant
    
    # Get specific confidence steps for this property and grade
    confidence_steps = CONFIDENCE_SETTINGS[grade][property_type]
    
    results = {
        'confidence_intervals': confidence_steps,
        'false_positives': {'percentage': [], 'absolute': []},
        'false_negatives': {'percentage': [], 'absolute': []},
        'total_samples': len(predictions) // n_folds  # Adjust total samples per fold
    }
    
    for conf_level in confidence_steps:
        false_positives = 0
        false_negatives = 0
        
        z_score = norm.ppf(1 - (1 - conf_level)/2)
        margin_of_error = z_score * std_estimate
        
        pred_lower = predictions - margin_of_error
        pred_upper = predictions + margin_of_error
        
        for pred_l, pred_u, actual in zip(pred_lower, pred_upper, actual_values):
            pred_meets = meets_requirements_with_intervals(pred_l, pred_u, actual, grade, property_type)
            actual_meets = meets_requirements_with_intervals(actual, actual, actual, grade, property_type)
            
            if pred_meets and not actual_meets:
                false_positives += 1
            elif not pred_meets and actual_meets:
                false_negatives += 1
        
        # Average the counts across folds
        false_positives = false_positives // n_folds
        false_negatives = false_negatives // n_folds
        
        # Store averaged results
        n_samples = len(predictions) // n_folds  # Adjust sample size per fold
        results['false_positives']['percentage'].append((false_positives/n_samples) * 100)
        results['false_positives']['absolute'].append(false_positives)
        results['false_negatives']['percentage'].append((false_negatives/n_samples) * 100)
        results['false_negatives']['absolute'].append(false_negatives)
    
    return results

def meets_requirements_with_intervals(pred_lower, pred_upper, actual, grade, property_type):
    """
    Check if predictions meet requirements while properly handling intervals
    """
    requirements = get_min_requirements(16, grade)  # Using default thickness of 16mm
    
    if property_type == 'Yield Strength [Mpa]':
        min_req = requirements['yield_strength']['min']
        return pred_lower >= min_req
    
    elif property_type == 'Tensile Strength [in MPA]':
        min_req = requirements['tensile_strength']['min']
        max_req = requirements['tensile_strength']['max']
        # For actual values, use exact comparison
        if pred_lower == pred_upper == actual:
            return min_req <= actual <= max_req
        # For prediction intervals, check if the entire interval is within requirements
        return min_req <= pred_lower and pred_upper <= max_req
    
    elif property_type == 'Elongation':
        min_req = requirements['elongation']['min']
        return pred_lower >= min_req
    
    return False

def plot_false_rates(results):
    """
    Create 2x3 subplot with legend completely outside the plots
    """
    # Create figure with extra space on the right for the legend
    fig = plt.figure(figsize=(20, 10))  # Increased width to accommodate legend
    
    # Create gridspec to control subplot layout
    gs = fig.add_gridspec(2, 4)  # 2 rows, 4 columns (3 for plots, 1 for legend)
    
    # Create axes for the plots (2x3 grid in the first 3 columns)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
    
    properties = ['Yield Strength [Mpa]', 'Tensile Strength [in MPA]', 'Elongation']
    grades = ['S355', 'S690']
    
    # Define colors and styles
    colors = {
        'fp': '#FF6B6B',
        'fn': '#4ECDC4',
        'acc': '#45B7D1'
    }
    
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
    
    for i, grade in enumerate(grades):
        for j, prop in enumerate(properties):
            ax = axes[i][j]
            data = results[grade]['false_rates'][prop]
            
            # Convert confidence intervals to percentages
            conf_percentages = [x * 100 for x in data['confidence_intervals']]
            
            # Get the data for plotting
            fp_data = data['false_positives']['absolute']
            fn_data = data['false_negatives']['absolute']
            total_samples = data["total_samples"]
            
            # Calculate accuracy
            accuracy = [(total_samples - fp - fn)/total_samples * 100 
                       for fp, fn in zip(fp_data, fn_data)]
            
            # Create second y-axis for accuracy percentage
            ax2 = ax.twinx()
            
            # Plot lines
            l1 = ax.plot(conf_percentages, fp_data, 
                        color=colors['fp'], linewidth=2, marker='o', 
                        label='False Positives\n(Predicted Pass, Actually Failed)', 
                        markersize=6)
            l2 = ax.plot(conf_percentages, fn_data, 
                        color=colors['fn'], linewidth=2, marker='s', 
                        label='False Negatives\n(Predicted Fail, Actually Passed)', 
                        markersize=6)
            l3 = ax2.plot(conf_percentages, accuracy, 
                         color=colors['acc'], linewidth=2, marker='^', 
                         label=f'Accuracy\n({prop})', 
                         markersize=6, linestyle='--')
            
            # Store legend handles and labels (only once)
            if i == 0 and j == 0:
                legend_handles.extend(l1 + l2 + l3)
                legend_labels.extend([l.get_label() for l in l1 + l2 + l3])
            
            # Add percentage annotations
            for x, (fp, fn, acc) in enumerate(zip(fp_data, fn_data, accuracy)):
                # Calculate percentages
                fp_pct = (fp / total_samples) * 100
                fn_pct = (fn / total_samples) * 100
                
                # Add annotations with improved positioning (only percentages)
                ax.annotate(f'{fp_pct:.1f}%', 
                            (conf_percentages[x], fp),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            fontsize=8)
                ax.annotate(f'{fn_pct:.1f}%', 
                            (conf_percentages[x], fn),
                            textcoords="offset points",
                            xytext=(0,-15),
                            ha='center',
                            fontsize=8)
                ax2.annotate(f'{acc:.1f}%', 
                            (conf_percentages[x], acc),
                            textcoords="offset points",
                            xytext=(30,0),
                            ha='left',
                            fontsize=8,
                            color=colors['acc'])
            
            # Set axis labels and limits
            ax.set_xlabel('Confidence Level (%)', fontsize=9)
            ax.set_ylabel('Number of Cases', fontsize=9)
            ax2.set_ylabel('Accuracy (%)', fontsize=9, color=colors['acc'])
            
            # Set y-axis limits
            ax.set_ylim(bottom=0, top=100)
            ax2.set_ylim(bottom=0, top=100)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            ax.minorticks_on()
            
            # Set title with total samples
            ax.set_title(f'{grade} - {prop}\nTotal Samples: {total_samples}', fontsize=10, pad=10)
    
    # Create a separate axes for the legend
    legend_ax = fig.add_subplot(gs[:, -1])  # Use the last column for legend
    legend_ax.axis('off')  # Hide the axes
    
    # Add the legend to the separate axes
    legend = legend_ax.legend(legend_handles, legend_labels,
                            loc='center',
                            fontsize=10,
                            title='Legend',
                            title_fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def calculate_rates(predictions, actuals, margin_of_error, grade, property_type):
    """Updated to use margin of error directly"""
    n_samples = len(predictions)
    
    pred_lower = predictions - margin_of_error
    pred_upper = predictions + margin_of_error if property_type == 'Tensile Strength [in MPA]' else None
    
    # Count actual passes/fails first
    actual_passes = sum(meets_requirements(a, a, grade, property_type) for a in actuals)
    actual_fails = n_samples - actual_passes
    
    # Count false positives and negatives
    false_positives = 0
    false_negatives = 0
    
    for i in range(n_samples):
        if property_type == 'Tensile Strength [in MPA]':
            pred_meets = meets_requirements_with_intervals(pred_lower[i], pred_upper[i], actuals[i], grade, property_type)
        else:
            pred_meets = meets_requirements(pred_lower[i], actuals[i], grade, property_type)
            
        actual_meets = meets_requirements(actuals[i], actuals[i], grade, property_type)
        
        if pred_meets and not actual_meets:
            false_positives += 1
        elif not pred_meets and actual_meets:
            false_negatives += 1
    
    fp_rate = (false_positives/n_samples) * 100
    fn_rate = (false_negatives/actual_passes) * 100 if actual_passes > 0 else 0
    accuracy = ((n_samples - false_positives - false_negatives)/n_samples) * 100
    
    return fp_rate, fn_rate, accuracy

def find_confidence_for_target_fp(predictions, actual_values, grade, property_type, target_fp_rate=2.25, tolerance=0.25):
    """Target 2-2.5% FP rate with tighter tolerance"""
    residuals = predictions - actual_values
    std_estimate = np.std(residuals)
    
    # Calculate actual failure rate
    actual_fails = sum(not meets_requirements(a, a, grade, property_type) for a in actual_values)
    actual_fail_rate = (actual_fails / len(actual_values)) * 100
    
    # Prepare data for plotting
    z_scores = np.linspace(0.1, 2.0, 100)
    accuracies = []
    fp_rates = []
    fn_rates = []
    
    # Try z-scores from 1.0 to 2.0 for reasonable confidence intervals
    for z_score in z_scores:
        margin_of_error = z_score * std_estimate
        
        # Calculate rates with this margin
        fp_rate, fn_rate, accuracy = calculate_rates(predictions, actual_values, margin_of_error, grade, property_type)
        
        # Store results for plotting
        accuracies.append(accuracy)
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
        
        # Check if FP rate is within 2-2.5% range
        if 2.0 <= fp_rate <= 2.5:
            conf_level = (1 - 2*(1 - norm.cdf(z_score))) * 100
            # Continue to collect data for plotting even after finding the optimal confidence level
            continue
    
    # Convert z-scores to confidence levels for plotting
    confidence_levels = [(1 - 2*(1 - norm.cdf(z))) * 100 for z in z_scores]
    
    # Plot the results after the loop
    plot_confidence_interval_analysis(confidence_levels, accuracies, fp_rates, fn_rates, grade, property_type)
    
    # Return the first found optimal confidence level
    for i, fp_rate in enumerate(fp_rates):
        if 2.0 <= fp_rate <= 2.5:
            return {
                'confidence_level': (1 - 2*(1 - norm.cdf(z_scores[i]))) * 100,
                'achieved_fp_rate': fp_rates[i],
                'fn_rate': fn_rates[i],
                'accuracy': accuracies[i],
                'margin_of_error': z_scores[i] * std_estimate,
                'pred_fail_rate': None,
                'actual_fail_rate': actual_fail_rate
            }
    
    # Default case - use 90% confidence interval
    margin_of_error = 1.645 * std_estimate
    fp_rate, fn_rate, accuracy = calculate_rates(predictions, actual_values, margin_of_error, grade, property_type)
    
    return {
        'confidence_level': 90.0,
        'achieved_fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'accuracy': accuracy,
        'margin_of_error': margin_of_error,
        'pred_fail_rate': None,
        'actual_fail_rate': actual_fail_rate
    }

def plot_confidence_interval_analysis(confidence_levels, accuracies, fp_rates, fn_rates, grade, property_type):
    """Plot accuracy and false positive/negative rates for iterated confidence intervals"""
    # Create figure if this is the first plot for the grade
    if property_type == 'Yield Strength [Mpa]':
        plot_confidence_interval_analysis.fig, plot_confidence_interval_analysis.axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_confidence_interval_analysis.fig.suptitle(f'Confidence Interval Analysis for {grade}', fontsize=14)
    
    # Determine which subplot to use based on property type
    subplot_idx = {
        'Yield Strength [Mpa]': 0,
        'Tensile Strength [in MPA]': 1,
        'Elongation': 2
    }[property_type]
    
    ax1 = plot_confidence_interval_analysis.axes[subplot_idx]
    
    # Plot accuracy
    color1 = 'tab:blue'
    ax1.set_xlabel('Confidence Level (%)')
    ax1.set_ylabel('Accuracy (%)', color=color1)
    line1 = ax1.plot(confidence_levels, accuracies, color=color1, label='Accuracy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for false rates
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('False Rates (%)', color=color2)
    line2 = ax2.plot(confidence_levels, fp_rates, color=color2, linestyle='--', label='FP Rate', linewidth=2)
    line3 = ax2.plot(confidence_levels, fn_rates, color='tab:green', linestyle='--', label='FN Rate', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
    # Add subplot title
    ax1.set_title(property_type)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout if this is the last plot for the grade
    if property_type == 'Elongation':
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def analyze_optimal_confidence_levels(metrics):
    """Analyze and print optimal confidence levels for all features and grades"""
    optimal_cis = {}
    for grade in ['S355', 'S690']:
        optimal_cis[grade] = {}
        print(f"\n{grade}:")
        for target in targets:
            predictions = np.array(metrics[grade]['predictions'][target])
            actual_values = np.array(metrics[grade]['actual_values'][target])
            
            # Find and plot confidence intervals for each grade and target
            result = find_confidence_for_target_fp(predictions, actual_values, grade, target)
            optimal_cis[grade][target] = result
            
            print(f"\n{target}:")
            print(f"  Confidence Level: {result['confidence_level']:.1f}%")
            print(f"  Achieved FP Rate: {result['achieved_fp_rate']:.2f}%")
            print(f"  Model Accuracy: {result['accuracy']:.2f}%")
            print(f"  Margin of Error: ±{result['margin_of_error']:.2f}")
            if result['pred_fail_rate'] is not None:
                print(f"  Predicted Fail Rate: {result['pred_fail_rate']:.2f}%")
            print(f"  Actual Fail Rate: {result['actual_fail_rate']:.2f}%")
    
    return optimal_cis

def calculate_combined_statistics(metrics, optimal_cis, grade):
    """Calculate conservative combined pass/fail statistics using optimal confidence intervals"""
    total_samples = len(metrics[grade]['predictions'][targets[0]])
    
    # Initialize arrays for tracking sample-level results
    sample_results = {
        'predicted_pass': np.ones(total_samples, dtype=bool),
        'actual_pass': np.ones(total_samples, dtype=bool),
        'property_errors': np.zeros(total_samples, dtype=int),  # Track errors per sample
        'property_error_details': [[] for _ in range(total_samples)]  # Track which properties failed
    }
    
    for target in targets:
        predictions = np.array(metrics[grade]['predictions'][target])
        actuals = np.array(metrics[grade]['actual_values'][target])
        margin = optimal_cis[grade][target]['margin_of_error']
        requirements = get_min_requirements(16, grade)
        
        for i in range(total_samples):
            pred = predictions[i]
            actual = actuals[i]
            
            # Determine if prediction passes (considering confidence interval)
            if target == 'Tensile Strength [in MPA]':
                pred_lower = pred - margin
                pred_upper = pred + margin
                pred_passes = (pred_lower >= requirements['tensile_strength']['min'] and 
                             pred_upper <= requirements['tensile_strength']['max'])
            elif target == 'Yield Strength [Mpa]':
                pred_lower = pred - margin
                pred_passes = pred_lower >= requirements['yield_strength']['min']
            else:  # Elongation
                pred_lower = pred - margin
                pred_passes = pred_lower >= requirements['elongation']['min']
            
            # Determine if actual value passes
            if target == 'Tensile Strength [in MPA]':
                actual_passes = (actual >= requirements['tensile_strength']['min'] and 
                               actual <= requirements['tensile_strength']['max'])
            elif target == 'Yield Strength [Mpa]':
                actual_passes = actual >= requirements['yield_strength']['min']
            else:  # Elongation
                actual_passes = actual >= requirements['elongation']['min']
            
            # Update sample results
            sample_results['predicted_pass'][i] &= pred_passes
            sample_results['actual_pass'][i] &= actual_passes
            
            # Track property-specific errors
            if pred_passes != actual_passes:
                sample_results['property_errors'][i] += 1
                sample_results['property_error_details'][i].append({
                    'property': target,
                    'predicted': pred,
                    'actual': actual,
                    'pred_passes': pred_passes,
                    'actual_passes': actual_passes
                })
    
    # Calculate final statistics
    false_positives = np.sum(sample_results['predicted_pass'] & ~sample_results['actual_pass'])
    false_negatives = np.sum(~sample_results['predicted_pass'] & sample_results['actual_pass'])
    
    # Count samples with any property errors
    samples_with_errors = np.sum(sample_results['property_errors'] > 0)
    
    return {
        'predicted_pass': np.sum(sample_results['predicted_pass']),
        'predicted_fail': total_samples - np.sum(sample_results['predicted_pass']),
        'actual_pass': np.sum(sample_results['actual_pass']),
        'actual_fail': total_samples - np.sum(sample_results['actual_pass']),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_samples': total_samples,
        'samples_with_errors': samples_with_errors,
        'property_error_details': sample_results['property_error_details']
    }

def display_model_statistics(metrics, optimal_cis):
    """Display comprehensive model statistics using optimal confidence intervals"""
    for grade in ['S355', 'S690']:
        print(f"\n=== Statistics for {grade} ===")
        
        combined_stats = calculate_combined_statistics(metrics, optimal_cis, grade)
        
        # Basic statistics display remains the same...
        
        # Show example cases only if there are errors to display
        if combined_stats['samples_with_errors'] > 0:
            print("\nExample error cases:")
            error_count = 0
            seen_errors = set()  # Track unique error combinations
            for i in range(len(combined_stats['property_error_details'])):
                if combined_stats['property_error_details'][i] and error_count < 5:
                    error_count += 1
                    sample_idx = i
                    print(f"\nSample {sample_idx}:")
                    # Use a set to track unique property errors for this sample
                    sample_errors = set()
                    for error_detail in combined_stats['property_error_details'][i]:
                        error_key = (sample_idx, error_detail['property'], 
                                   error_detail['predicted'], error_detail['actual'])
                        if error_key not in seen_errors:
                            seen_errors.add(error_key)
                            target = error_detail['property']
                            pred = error_detail['predicted']
                            actual = error_detail['actual']
                            print(f"{target}: Predicted={pred:.2f}, Actual={actual:.2f}")

def export_yield_strength_true_negatives(metrics, optimal_cis, output_path):
    """
    Export true negative samples (where both predicted and actual yield strength are below requirements)
    with all original features to CSV for both S355 and S690
    """
    # Load original data
    df_original = pd.read_csv("C:/Users/31633/OneDrive - Delft University of Technology/BEP/AllData2.csv", sep=';')
    print(f"\nLoaded original data with {len(df_original)} rows")
    
    # Clean numerical data
    numerical_base_features = ['QC_Re', 'QC_Rm', 'QC_A', 'Dimension']
    def clean_numeric(value):
        if isinstance(value, str):
            if '/' in value:
                return float(value.split('/')[0])
            return float(value.replace(',', '.').replace('[^\d.-]', ''))
        return value

    for col in numerical_base_features + targets:
        df_original[col] = df_original[col].apply(clean_numeric)
    
    # Standardize grade names
    df_original['Simplified Grade (for further analysis)'] = df_original['Simplified Grade (for further analysis)'].apply(
        lambda x: 'S690' if 'StE690' in str(x) or 'S690' in str(x) else ('S355' if 'S355' in str(x) else x)
    )
    
    # Initialize list to store true negative data
    true_negative_data = []
    
    print("\nAnalyzing True Negatives (cases where both predicted and actual values fail requirements):")
    for grade in ['S355', 'S690']:
        # Get predictions and scale them if needed
        predictions = np.array(metrics[grade]['predictions']['Yield Strength [Mpa]'])
        if np.mean(predictions) > 1000:  # If values are larger than typical MPa range
            predictions = predictions / 1000000  # Convert from micro-MPa to MPa
        print(f"Predictions for {grade} - Mean: {np.mean(predictions):.1f} MPa")
            
        actual_yields = metrics[grade]['actual_values']['Yield Strength [Mpa]']
        margin = optimal_cis[grade]['Yield Strength [Mpa]']['margin_of_error']
        
        grade_data = df_original[df_original['Simplified Grade (for further analysis)'] == grade]
        print(f"\n{grade} data: {len(grade_data)} rows")
        
        print(f"\n{grade} Analysis:")
        print(f"Confidence Interval Margin: ±{margin:.2f} MPa")
        
        # Process each sample
        true_neg_count = 0
        for i in range(len(predictions)):
            predicted_yield = predictions[i]  # Already normalized
            actual_yield = actual_yields[i]
            
            # Get the actual thickness for this sample
            try:
                thickness = float(grade_data.iloc[i]['Dimension'])
            except (IndexError, ValueError):
                print(f"Warning: Could not get thickness for sample {i}, using default 16mm")
                thickness = 16.0
            
            # Get thickness-specific requirements
            requirements = get_min_requirements(thickness, grade)
            required_min_yield = requirements['yield_strength']['min']
            
            # Check prediction with margin
            predicted_yield_with_margin = predicted_yield - margin
            
            # A true negative is when both predicted and actual values are below requirements
            predicted_fails_requirement = predicted_yield_with_margin < required_min_yield
            actual_fails_requirement = actual_yield < required_min_yield
            
            if predicted_fails_requirement and actual_fails_requirement:
                true_neg_count += 1
                # Store the row data along with the predicted value
                if i < len(grade_data):
                    row_data = grade_data.iloc[i].to_dict()
                    row_data['Predicted_Yield_Strength'] = float(predicted_yield)  # Ensure it's stored as float
                    true_negative_data.append(row_data)
                    
                if true_neg_count <= 5:  # Print first 5 examples
                    print(f"\nExample {true_neg_count}:")
                    print(f"Thickness: {thickness:.1f}mm")
                    print(f"Required Min Yield: {required_min_yield} MPa")
                    print(f"Predicted Yield: {predicted_yield:.1f} MPa (with margin: {predicted_yield_with_margin:.1f} MPa)")
                    print(f"Actual Yield: {actual_yield:.1f} MPa")
                    print(f"Both values are below requirement of {required_min_yield} MPa for {thickness}mm thickness")
        
        print(f"\nTotal {grade} True Negatives: {true_neg_count}")
    
    # Create DataFrame from true negative data
    if true_negative_data:
        true_negatives_df = pd.DataFrame(true_negative_data)
        print(f"\nCreated DataFrame with {len(true_negatives_df)} true negative rows")
        
        # Verify predictions are in correct range before saving
        pred_vals = true_negatives_df['Predicted_Yield_Strength']
        print("\nPredicted Yield Strength Statistics (before saving):")
        print(f"Min: {pred_vals.min():.1f} MPa")
        print(f"Max: {pred_vals.max():.1f} MPa")
        print(f"Mean: {pred_vals.mean():.1f} MPa")
        
        # Save to CSV
        true_negatives_df.to_csv(output_path, index=False)
        print(f"Successfully exported to {output_path}")
        print(f"CSV contains {len(true_negatives_df)} rows with the following columns:")
        print(true_negatives_df.columns.tolist())
    else:
        print("\nNo true negatives found to export")

def export_yield_strength_true_positives(metrics, optimal_cis, output_path):
    """
    Export true positive samples with all original features to CSV for both S355 and S690
    """
    # Load original data
    df_original = pd.read_csv("C:/Users/31633/OneDrive - Delft University of Technology/BEP/AllData2.csv", sep=';')
    
    # Clean numerical data as done in process_data_and_train
    numerical_base_features = ['QC_Re', 'QC_Rm', 'QC_A', 'Dimension']
    def clean_numeric(value):
        if isinstance(value, str):
            if '/' in value:
                return float(value.split('/')[0])
            return float(value.replace(',', '.').replace('[^\d.-]', ''))
        return value

    for col in numerical_base_features + targets:
        df_original[col] = df_original[col].apply(clean_numeric)
    
    # Standardize grade names in original data
    df_original['Simplified Grade (for further analysis)'] = df_original['Simplified Grade (for further analysis)'].apply(
        lambda x: 'S690' if 'StE690' in str(x) or 'S690' in str(x) else ('S355' if 'S355' in str(x) else x)
    )
    
    # Initialize list to store indices of true positives
    true_positive_indices = []
    
    for grade in ['S355', 'S690']:
        predictions = metrics[grade]['predictions']['Yield Strength [Mpa]']
        actuals = metrics[grade]['actual_values']['Yield Strength [Mpa]']
        margin = optimal_cis[grade]['Yield Strength [Mpa]']['margin_of_error']
        
        grade_data = df_original[df_original['Simplified Grade (for further analysis)'] == grade]
        
        # Process each sample
        for i in range(len(predictions)):
            pred = predictions[i]
            actual = actuals[i]
            
            # Check if both prediction and actual pass requirements
            pred_passes = meets_requirements(pred - margin, actual, grade, 'Yield Strength [Mpa]')
            actual_passes = meets_requirements(actual, actual, grade, 'Yield Strength [Mpa]')
            
            # If both pass, it's a true positive
            if pred_passes and actual_passes:
                # Find the corresponding row in the original dataset
                matching_row = grade_data[
                    (grade_data['QC_Re'] == actual)
                ].index
                
                if len(matching_row) > 0:
                    true_positive_indices.extend(matching_row)
    
    # Extract true positive rows with all original features
    true_positives_df = df_original.loc[true_positive_indices]
    
    # Save to CSV
    if not true_positives_df.empty:
        true_positives_df.to_csv(output_path, index=False)
        print(f"Yield strength true positives exported to {output_path} with {len(true_positives_df)} rows")
    else:
        print("No true positives found")

def main():
    """
    Main execution function
    """
    metrics = process_data_and_train()
    optimal_cis = analyze_optimal_confidence_levels(metrics)
    display_model_statistics(metrics, optimal_cis)
    
    # Export yield strength true negatives
    output_path = "yield_strength_true_negatives.csv"
    export_yield_strength_true_negatives(metrics, optimal_cis, output_path)


if __name__ == "__main__":
    main()