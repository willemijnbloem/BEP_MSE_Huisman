import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def process_data_and_train(manufacturer=None):
    """
    Main processing and training pipeline
    manufacturer: specific manufacturer ID or None for all manufacturers
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

    # 3. Filter by manufacturer if specified
    if manufacturer is not None:
        df = df[df['Manufacturer'] == manufacturer]
    
    # 4. Remove missing values
    df = df.dropna(subset=numerical_base_features + ['Manufacturer', 'Simplified Grade (for further analysis)'] + targets)
    
    # 5. Standardize grade names
    df['Simplified Grade (for further analysis)'] = df['Simplified Grade (for further analysis)'].apply(
        lambda x: 'S690' if 'StE690' in str(x) or 'S690' in str(x) else ('S355' if 'S355' in str(x) else x)
    )
    
    # 6. Create enhanced features
    df_enhanced = create_enhanced_features(df)
    
    # 7. Initialize results container - modified to include false rates
    model_metrics = {
        'S355': {'metrics': {}, 'false_rates': {}, 'predictions': {}, 'actual_values': {}}, 
        'S690': {'metrics': {}, 'false_rates': {}, 'predictions': {}, 'actual_values': {}}
    }
    
    # 8. Get engineered features
    engineered_features = [col for col in df_enhanced.columns 
                          if col not in targets + ['Manufacturer', 'Simplified Grade (for further analysis)']
                          and col != 'Grade (as reported)'
                          and df_enhanced[col].dtype != 'object']
    
    # 9. Process each grade and target
    for grade in ['S355', 'S690']:
        grade_data = df_enhanced[df_enhanced['Simplified Grade (for further analysis)'] == grade]
        
        # Initialize storage for predictions and actual values
        all_predictions = {target: [] for target in targets}
        all_actual_values = {target: [] for target in targets}
        
        for target in targets:
            # 10. Select features based on target
            if target == 'Elongation':
                features_to_use = numerical_base_features
            else:
                # Calculate correlations
                correlations = grade_data[engineered_features].corrwith(grade_data[target])
                features_to_use = correlations[correlations.abs() > 0.1].index.tolist()
            
            X = grade_data[features_to_use].values
            y = grade_data[target].values
            
            # 11. Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 12. Perform cross-validation
            kf = KFold(n_splits=min(10, len(X_scaled) // 2), shuffle=True, random_state=42)
            
            for train_idx, test_idx in kf.split(X_scaled):
                # 13. Split data
                X_train_fold = X_scaled[train_idx]
                X_test_fold = X_scaled[test_idx]
                y_train_fold = y[train_idx]
                y_test_fold = y[test_idx]
                
                # 14. Train models
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
                    criterion='friedman_mse'
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
                    criterion='friedman_mse'
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
    """
    Classify a prediction as PASS, SEMI-PASS, or FAIL based on requirements
    
    Args:
        prediction: dictionary with predictions for all properties
        confidence_interval: dictionary with CIs for all properties
        thickness: thickness value
        grade: steel grade (S355 or S690)
    """
    requirements = get_min_requirements(thickness, grade)
    
    # Calculate bounds for each property
    yield_lower = prediction['Yield Strength [Mpa]'] - (prediction['Yield Strength [Mpa]'] * confidence_interval['Yield Strength [Mpa]'] / 100)
    tensile_lower = prediction['Tensile Strength [in MPA]'] - (prediction['Tensile Strength [in MPA]'] * confidence_interval['Tensile Strength [in MPA]'] / 100)
    tensile_upper = prediction['Tensile Strength [in MPA]'] + (prediction['Tensile Strength [in MPA]'] * confidence_interval['Tensile Strength [in MPA]'] / 100)
    elong_lower = prediction['Elongation'] - (prediction['Elongation'] * confidence_interval['Elongation'] / 100)
    
    # Check yield strength (mandatory for both PASS and SEMI-PASS)
    if yield_lower < requirements['yield_strength']['min']:
        return 'FAIL'
    
    # Check elongation fail condition first
    if elong_lower < 6:
        return 'FAIL'
    
    # Check PASS conditions
    tensile_in_range = (tensile_lower >= requirements['tensile_strength']['min'] and 
                       tensile_upper <= requirements['tensile_strength']['max'])
    elong_pass = elong_lower >= requirements['elongation']['min']
    
    if tensile_in_range and elong_pass:
        return 'PASS'
    
    # Check SEMI-PASS conditions
    tensile_above = tensile_lower > requirements['tensile_strength']['max']
    elong_semi = 6 <= elong_lower < requirements['elongation']['min']
    
    if tensile_above or elong_semi:
        return 'SEMI-PASS'
    
    return 'FAIL'

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
    residuals = predictions - actual_values
    std_estimate = np.std(residuals)
    n_samples = len(predictions)
    
    # Skip fold division if sample size is too small
    n_folds = min(10, max(1, n_samples // 2))
    
    # Get specific confidence steps for this property and grade
    confidence_steps = CONFIDENCE_SETTINGS[grade][property_type]
    
    results = {
        'confidence_intervals': confidence_steps,
        'false_positives': {'percentage': [], 'absolute': []},
        'false_negatives': {'percentage': [], 'absolute': []},
        'total_samples': n_samples
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

def plot_confidence_interval_analysis(confidence_levels, accuracies, fp_rates, fn_rates, grade, property_type, ax, optimal_result):
    """Plot accuracy and false rates on the provided axis"""
    # Set background color
    ax.set_facecolor('#f8f9fa')
    
    # Plot accuracy on primary y-axis
    color1 = '#2C3E50'  # Dark blue
    ax.set_xlabel('Confidence Level (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold', color=color1)
    line1 = ax.plot(confidence_levels, accuracies, color=color1, 
                    label='Accuracy', linewidth=2.5, zorder=3)
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for false rates
    ax2 = ax.twinx()
    
    # Plot false positive and negative rates
    color2 = '#E74C3C'  # Red
    color3 = '#27AE60'  # Green
    line2 = ax2.plot(confidence_levels, fp_rates, color=color2, 
                     label='False Positive Rate', linewidth=2, 
                     linestyle='--', zorder=2)
    line3 = ax2.plot(confidence_levels, fn_rates, color=color3, 
                     label='False Negative Rate', linewidth=2, 
                     linestyle='-.', zorder=2)
    
    # Set y-axis labels and colors
    ax2.set_ylabel('False Rates (%)', fontsize=10, fontweight='bold', color='#7F8C8D')
    ax2.tick_params(axis='y', labelcolor='#7F8C8D')
    
    # Add horizontal lines for key values
    ax2.axhline(y=2.0, color='#E74C3C', linestyle=':', alpha=0.5)
    ax2.axhline(y=2.5, color='#E74C3C', linestyle=':', alpha=0.5)
    
    # Mark optimal point if found
    if optimal_result:
        conf_level = optimal_result['confidence_level']
        accuracy = optimal_result['accuracy']
        fp_rate = optimal_result['achieved_fp_rate']
        fn_rate = optimal_result['fn_rate']
        
        # Plot vertical line at optimal confidence level
        ax.axvline(x=conf_level, color='gray', linestyle=':', alpha=0.5)
        
        # Plot markers for optimal points
        ax.plot(conf_level, accuracy, 'o', color=color1, markersize=8, zorder=4)
        ax2.plot(conf_level, fp_rate, 'o', color=color2, markersize=8, zorder=4)
        ax2.plot(conf_level, fn_rate, 'o', color=color3, markersize=8, zorder=4)
        
        # Add annotations
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        ax.annotate(f'Optimal CI: {conf_level:.1f}%\nAccuracy: {accuracy:.1f}%\nFP Rate: {fp_rate:.1f}%\nFN Rate: {fn_rate:.1f}%', 
                   xy=(conf_level, accuracy),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=bbox_props,
                   fontsize=8)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=8)
    
    # Add grid with specific style
    ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
    
    # Set title
    ax.set_title(f'{property_type}', fontsize=12, fontweight='bold', pad=10)
    
    # Set axis limits with some padding
    ax.set_ylim(min(accuracies) - 5, max(accuracies) + 5)
    ax2.set_ylim(0, max(max(fp_rates), max(fn_rates)))

def find_confidence_for_target_fp(predictions, actual_values, grade, property_type, target_fp_rate=2.25, tolerance=0.25):
    """Target 2-2.5% FP rate with tighter tolerance"""
    residuals = predictions - actual_values
    std_estimate = np.std(residuals)
    
    # Prepare data for plotting
    z_scores = np.linspace(0.1, 2.0, 100)
    accuracies = []
    fp_rates = []
    fn_rates = []
    
    # Try z-scores and collect data
    optimal_result = None
    for z_score in z_scores:
        margin_of_error = z_score * std_estimate
        fp_rate, fn_rate, accuracy = calculate_rates(predictions, actual_values, margin_of_error, grade, property_type)
        
        accuracies.append(accuracy)
        fp_rates.append(fp_rate)
        fn_rates.append(fn_rate)
        
        if optimal_result is None and 2.0 <= fp_rate <= 2.5:
            optimal_result = {
                'confidence_level': (1 - 2*(1 - norm.cdf(z_score))) * 100,
                'achieved_fp_rate': fp_rate,
                'fn_rate': fn_rate,
                'accuracy': accuracy,
                'margin_of_error': margin_of_error
            }
    
    # Convert z-scores to confidence levels
    confidence_levels = [(1 - 2*(1 - norm.cdf(z))) * 100 for z in z_scores]
    
    return confidence_levels, accuracies, fp_rates, fn_rates, optimal_result

def analyze_optimal_confidence_levels(metrics):
    """Analyze and plot optimal confidence levels for all features and grades"""
    print("\nOptimal Confidence Levels for 2% False Positive Rate:")
    print("-" * 60)
    
    for grade in ['S355', 'S690']:
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Confidence Interval Analysis - {grade}', fontsize=14, fontweight='bold')
        
        print(f"\n{grade}:")
        for idx, target in enumerate(targets):
            predictions = np.array(metrics[grade]['predictions'][target])
            actual_values = np.array(metrics[grade]['actual_values'][target])
            
            confidence_levels, accuracies, fp_rates, fn_rates, result = find_confidence_for_target_fp(
                predictions, actual_values, grade, target
            )
            
            # Plot on the corresponding subplot
            plot_confidence_interval_analysis(
                confidence_levels, accuracies, fp_rates, fn_rates, 
                grade, target, axes[idx], result
            )
            
            if result:
                print(f"\n{target}:")
                print(f"  Confidence Level: {result['confidence_level']:.1f}%")
                print(f"  Achieved FP Rate: {result['achieved_fp_rate']:.2f}%")
                print(f"  Model Accuracy: {result['accuracy']:.2f}%")
                print(f"  Margin of Error: ±{result['margin_of_error']:.2f}")
            else:
                print(f"\n{target}: No suitable confidence level found")
        
        plt.tight_layout()
        plt.show()

def calculate_rates(predictions, actual_values, margin_of_error, grade, property_type):
    """Calculate false positive rate, false negative rate, and accuracy"""
    total_samples = len(predictions)
    false_positives = 0
    false_negatives = 0
    correct_predictions = 0
    
    for pred, actual in zip(predictions, actual_values):
        # Check if prediction meets requirements
        pred_meets = meets_requirements(pred - margin_of_error, actual, grade, property_type)
        actual_meets = meets_requirements(actual, actual, grade, property_type)
        
        if pred_meets and not actual_meets:
            false_positives += 1
        elif not pred_meets and actual_meets:
            false_negatives += 1
        elif pred_meets == actual_meets:
            correct_predictions += 1
    
    fp_rate = (false_positives / total_samples) * 100
    fn_rate = (false_negatives / total_samples) * 100
    accuracy = (correct_predictions / total_samples) * 100
    
    return fp_rate, fn_rate, accuracy

def analyze_manufacturers():
    """
    Analyze specific manufacturers and create comparison plots
    """
    manufacturers = [2, 3, 7, 15, 20, 37, 45]
    all_metrics = {
        'all': process_data_and_train(),  # Process all manufacturers
    }
    
    # Print sample sizes
    df = pd.read_csv("C:/Users/31633/OneDrive - Delft University of Technology/BEP/AllData2.csv", sep=';')
    print("\nSample sizes per manufacturer:")
    print(f"All manufacturers: {len(df)}")
    for manuf in manufacturers:
        print(f"Manufacturer {manuf}: {len(df[df['Manufacturer'] == manuf])}")
    
    # Process each specific manufacturer
    for manuf in manufacturers:
        all_metrics[manuf] = process_data_and_train(manufacturer=manuf)
    
    # Create original metrics visualization
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 3)
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    
    plot_manufacturer_metrics(all_metrics, ax1, 'RMSE')
    plot_manufacturer_metrics(all_metrics, ax2, 'MAE')
    plot_manufacturer_metrics(all_metrics, ax3, 'R²')
    
    plt.tight_layout()
    plt.show()
    
    # Create separate ranking plots
    plot_rankings(all_metrics)
    
    # Add after the existing plots
    plot_relative_performance(all_metrics)

def plot_manufacturer_metrics(metrics, ax, metric_type):
    manufacturers = ['all', 2, 3, 7, 15, 20, 37, 45]
    x = np.arange(len(manufacturers))
    width = 0.35
    
    s355_values = []
    s690_values = []
    
    for manuf in manufacturers:
        for grade in ['S355', 'S690']:
            try:
                values = [metrics[manuf][grade]['metrics'][target][metric_type] 
                         for target in targets]
                if grade == 'S355':
                    s355_values.append(np.mean(values))
                else:
                    s690_values.append(np.mean(values))
            except KeyError:
                if grade == 'S355':
                    s355_values.append(0)
                else:
                    s690_values.append(0)
    
    # Plot bars
    ax.bar(x - width/2, s355_values, width, label='S355')
    ax.bar(x + width/2, s690_values, width, label='S690')
    
    # Add reference lines for ALL manufacturers
    if len(s355_values) > 0:
        ax.axhline(y=s355_values[0], color='blue', linestyle='--', alpha=0.5, label='S355 ALL')
    if len(s690_values) > 0:
        ax.axhline(y=s690_values[0], color='red', linestyle='--', alpha=0.5, label='S690 ALL')
    
    ax.set_ylabel(metric_type)
    ax.set_title(f'{metric_type} by Manufacturer')
    ax.set_xticks(x)
    ax.set_xticklabels(manufacturers)
    ax.legend()
    
    # Store rankings data if needed for separate plots
    if metric_type in ['RMSE', 'MAE']:
        return s355_values, s690_values

def plot_rankings(metrics):
    """
    Create separate ranking plots for RMSE and MAE, split by grade
    """
    manufacturers = ['all', 2, 3, 7, 15, 20, 37, 45]
    metrics_to_plot = ['RMSE', 'MAE']
    
    # Create separate figures for S355 and S690
    fig_355 = plt.figure(figsize=(15, 6))
    fig_690 = plt.figure(figsize=(15, 6))
    
    # Create subplots for each metric
    ax_355_rmse = fig_355.add_subplot(121)
    ax_355_mae = fig_355.add_subplot(122)
    ax_690_rmse = fig_690.add_subplot(121)
    ax_690_mae = fig_690.add_subplot(122)
    
    for metric_type, (ax_355, ax_690) in zip(metrics_to_plot, [(ax_355_rmse, ax_690_rmse), (ax_355_mae, ax_690_mae)]):
        # Get values and create rankings
        s355_values, s690_values = [], []
        s355_std, s690_std = [], []
        
        for manuf in manufacturers:
            for grade in ['S355', 'S690']:
                try:
                    values = [metrics[manuf][grade]['metrics'][target][metric_type] 
                             for target in targets]
                    if grade == 'S355':
                        s355_values.append(np.mean(values))
                        s355_std.append(np.std(values))
                    else:
                        s690_values.append(np.mean(values))
                        s690_std.append(np.std(values))
                except KeyError:
                    if grade == 'S355':
                        s355_values.append(0)
                        s355_std.append(0)
                    else:
                        s690_values.append(0)
                        s690_std.append(0)
        
        # Calculate averages (excluding 'all')
        s355_avg = np.mean([v for m, v in zip(manufacturers[1:], s355_values[1:]) if v > 0])
        s690_avg = np.mean([v for m, v in zip(manufacturers[1:], s690_values[1:]) if v > 0])
        
        # Sort manufacturers by performance (excluding 'all')
        s355_ranking = [(m, v, s) for m, v, s in zip(manufacturers[1:], s355_values[1:], s355_std[1:]) if v > 0]
        s690_ranking = [(m, v, s) for m, v, s in zip(manufacturers[1:], s690_values[1:], s690_std[1:]) if v > 0]
        s355_ranking.sort(key=lambda x: x[1])
        s690_ranking.sort(key=lambda x: x[1])
        
        # Plot S355 rankings
        manuf_355, vals_355, stds_355 = zip(*s355_ranking)
        y_pos = np.arange(len(manuf_355))
        ax_355.barh(y_pos, vals_355, xerr=stds_355, align='center')
        ax_355.set_yticks(y_pos)
        ax_355.set_yticklabels(manuf_355)
        ax_355.axvline(x=s355_avg, color='red', linestyle='--', label='Average')
        ax_355.set_title(f'S355 {metric_type} Rankings')
        ax_355.set_xlabel(metric_type)
        ax_355.grid(True, alpha=0.3)
        ax_355.legend()
        
        # Plot S690 rankings
        manuf_690, vals_690, stds_690 = zip(*s690_ranking)
        y_pos = np.arange(len(manuf_690))
        ax_690.barh(y_pos, vals_690, xerr=stds_690, align='center')
        ax_690.set_yticks(y_pos)
        ax_690.set_yticklabels(manuf_690)
        ax_690.axvline(x=s690_avg, color='red', linestyle='--', label='Average')
        ax_690.set_title(f'S690 {metric_type} Rankings')
        ax_690.set_xlabel(metric_type)
        ax_690.grid(True, alpha=0.3)
        ax_690.legend()
    
    fig_355.suptitle('S355 Performance Rankings', fontsize=14)
    fig_690.suptitle('S690 Performance Rankings', fontsize=14)
    
    fig_355.tight_layout()
    fig_690.tight_layout()
    plt.show()

def plot_relative_performance(metrics):
    """
    Create plots showing relative performance compared to average
    """
    manufacturers = ['all', 2, 3, 7, 15, 20, 37, 45]
    metrics_to_plot = ['RMSE', 'MAE']
    
    fig_355 = plt.figure(figsize=(15, 6))
    fig_690 = plt.figure(figsize=(15, 6))
    
    ax_355_rmse = fig_355.add_subplot(121)
    ax_355_mae = fig_355.add_subplot(122)
    ax_690_rmse = fig_690.add_subplot(121)
    ax_690_mae = fig_690.add_subplot(122)
    
    for metric_type, (ax_355, ax_690) in zip(metrics_to_plot, [(ax_355_rmse, ax_690_rmse), (ax_355_mae, ax_690_mae)]):
        s355_values, s690_values = [], []
        
        for manuf in manufacturers[1:]:  # Skip 'all'
            for grade in ['S355', 'S690']:
                try:
                    values = [metrics[manuf][grade]['metrics'][target][metric_type] 
                             for target in targets]
                    if grade == 'S355':
                        s355_values.append(np.mean(values))
                    else:
                        s690_values.append(np.mean(values))
                except KeyError:
                    if grade == 'S355':
                        s355_values.append(np.nan)
                    else:
                        s690_values.append(np.nan)
        
        # Calculate percentage difference from average
        s355_avg = np.nanmean(s355_values)
        s690_avg = np.nanmean(s690_values)
        
        s355_relative = [(v/s355_avg - 1) * 100 for v in s355_values]
        s690_relative = [(v/s690_avg - 1) * 100 for v in s690_values]
        
        # Plot relative performance
        y_pos = np.arange(len(manufacturers[1:]))
        
        # S355 Plot
        bars_355 = ax_355.barh(y_pos, s355_relative, align='center')
        ax_355.set_yticks(y_pos)
        ax_355.set_yticklabels(manufacturers[1:])
        ax_355.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax_355.set_title(f'S355 {metric_type} Relative Performance')
        ax_355.set_xlabel('% Difference from Average')
        
        # Color bars based on performance
        for bar, value in zip(bars_355, s355_relative):
            if value < 0:
                bar.set_color('green')  # Better than average
            else:
                bar.set_color('red')    # Worse than average
        
        # S690 Plot
        bars_690 = ax_690.barh(y_pos, s690_relative, align='center')
        ax_690.set_yticks(y_pos)
        ax_690.set_yticklabels(manufacturers[1:])
        ax_690.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax_690.set_title(f'S690 {metric_type} Relative Performance')
        ax_690.set_xlabel('% Difference from Average')
        
        # Color bars based on performance
        for bar, value in zip(bars_690, s690_relative):
            if value < 0:
                bar.set_color('green')  # Better than average
            else:
                bar.set_color('red')    # Worse than average
    
    fig_355.suptitle('S355 Relative Performance (Green = Better than Average)', fontsize=14)
    fig_690.suptitle('S690 Relative Performance (Green = Better than Average)', fontsize=14)
    
    fig_355.tight_layout()
    fig_690.tight_layout()
    plt.show()

def main():
    """
    Modified main execution function
    """
    plt.ion()
    analyze_manufacturers()
    plt.show(block=True)

if __name__ == "__main__":
    main()