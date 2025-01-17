import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define targets and mappings
targets = ['Yield Strength [Mpa]', 'Tensile Strength [in MPA]', 'Elongation', 'ImpactAvg']

COLUMN_MAPPINGS = {
    'Yield Strength [Mpa]': 'QC_Re',
    'Tensile Strength [in MPA]': 'QC_Rm',
    'Elongation': 'QC_A',
    'ImpactAvg': 'QC_ImpactAvg'
}

# Define specific confidence intervals for each property and grade
CONFIDENCE_SETTINGS = {
    'S355': {
        'Yield Strength [Mpa]': np.linspace(0.01, 0.99, 10),
        'Tensile Strength [in MPA]': np.linspace(0.01, 0.99, 10),
        'Elongation': np.linspace(0.01, 0.99, 10),
        'ImpactAvg': np.linspace(0.01, 0.99, 10)
    },
    'S690': {
        'Yield Strength [Mpa]': np.linspace(0.01, 0.99, 10),
        'Tensile Strength [in MPA]': np.linspace(0.01, 0.99, 10),
        'Elongation': np.linspace(0.01, 0.99, 10),
        'ImpactAvg': np.linspace(0.01, 0.99, 10)
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
    
    # One-hot encode temperature values with cumulative effect
    # First convert temperature to Celsius if it's in Kelvin
    enhanced['QC_temp_C'] = enhanced['QC_temp'].apply(lambda x: x - 273.15 if x > 100 else x)
    
    # Create cumulative one-hot columns for each standard temperature
    valid_temps = sorted([-60, -50, -40, -20, -10, 0])  # Sort temperatures from coldest to warmest
    for i, temp in enumerate(valid_temps):
        # For each temperature, include all tests at this temperature OR colder
        enhanced[f'Temp_{temp}C'] = (enhanced['QC_temp_C'] <= temp).astype(int)
    
    # Drop the original temperature columns
    enhanced = enhanced.drop(['QC_temp', 'QC_temp_C'], axis=1)
    
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

def create_enhanced_features_original(df):
    """Original feature creation with simple one-hot encoding"""
    enhanced = df.copy()
    eps = 1e-10
    
    # Standard one-hot encoding
    enhanced['QC_temp_C'] = enhanced['QC_temp'].apply(lambda x: x - 273.15 if x > 100 else x)
    valid_temps = [-60, -50, -40, -20, -10, 0]
    for temp in valid_temps:
        enhanced[f'Temp_{temp}C'] = (enhanced['QC_temp_C'] == temp).astype(int)
    
    enhanced = enhanced.drop(['QC_temp', 'QC_temp_C'], axis=1)
    # ... rest of feature engineering ...
    return enhanced

def calculate_confidence_intervals(predictions, actual_values):
    """Calculate different confidence intervals from 0.1 to 0.9"""
    confidence_levels = np.linspace(0.1, 0.9, 10)  # 10 steps from 0.1 to 0.9
    intervals = {}
    
    for conf in confidence_levels:
        z_score = norm.ppf((1 + conf) / 2)  # Two-tailed
        std_dev = np.std(predictions)
        margin = z_score * std_dev
        
        # Calculate accuracy for this interval
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        within_ci = np.sum((actual_values >= lower_bound) & (actual_values <= upper_bound))
        accuracy = (within_ci / len(predictions)) * 100
        
        intervals[f"{conf:.1f}"] = {
            "margin": margin,
            "accuracy": accuracy
        }
    
    return intervals

def clean_numeric(value):
    """Clean numeric values, handling special cases"""
    if pd.isna(value):  # Handle NaN values
        return np.nan
        
    if isinstance(value, (int, float)):  # Already numeric
        return float(value)
        
    if isinstance(value, str):
        try:
            # Remove trailing spaces
            value = value.strip()
            
            # Skip non-numeric values like 'L' or 'Q345D'
            if value.isalpha():
                return np.nan
            
            # Handle values with multiple dots
            if value.count('.') > 1:
                # Replace all dots except the last one
                parts = value.split('.')
                value = parts[0] + '.' + ''.join(parts[1:])
            
            # Handle different types of inequality symbols
            if any(symbol in value for symbol in ['>', '<', '≥', '＞']):
                # Extract numeric part and remove any trailing spaces
                numeric_part = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(numeric_part)
            
            # Handle regular numeric values
            return float(value.replace(',', '.'))
            
        except ValueError as e:
            print(f"Warning: Could not convert value '{value}' to float")
            return np.nan
            
    return np.nan

def process_data_and_train():
    """Main processing and training pipeline"""
    # Initialize model metrics dictionary
    model_metrics = {grade: {'predictions': {}, 'actual_values': {}} for grade in ['S355', 'S690']}
    
    # 1. Load and prepare data
    df = pd.read_csv("C:/Users/31633/OneDrive - Delft University of Technology/BEP/AllData2.csv", sep=';')
    
    # Process impact test data
    df = expand_impact_rows(df)
    
    # 2. Clean numerical data
    numerical_base_features = ['QC_Re', 'QC_Rm', 'QC_A', 'Dimension', 'QC_temp']
    critical_columns = numerical_base_features + targets
    
    # Clean data silently
    for col in critical_columns:
        df[col] = df[col].apply(clean_numeric)

    # 3. Remove rows with NaN in critical columns
    df = df.dropna(subset=critical_columns)
    
    # 4. Standardize grade names
    df['Simplified Grade (for further analysis)'] = df['Simplified Grade (for further analysis)'].apply(
        lambda x: 'S690' if 'StE690' in str(x) or 'S690' in str(x) else ('S355' if 'S355' in str(x) else x)
    )
    
    # Add comparison of encoding methods
    print("\nComparing temperature encoding methods...")
    comparison_results = compare_encoding_methods(df)
    print_comparison_results(comparison_results)
    
    # Continue with the rest of the processing...
    # Create enhanced features
    df_enhanced = create_enhanced_features(df)
    
    # Continue with feature selection
    engineered_features = [col for col in df_enhanced.columns 
                          if col not in targets + ['Manufacturer', 'Simplified Grade (for further analysis)']
                          and col != 'Grade (as reported)'
                          and df_enhanced[col].dtype != 'object']
    
    for grade in ['S355', 'S690']:
        grade_data = df_enhanced[df_enhanced['Simplified Grade (for further analysis)'] == grade]
        
        correlations = grade_data[engineered_features].corrwith(grade_data['ImpactAvg'])
        features_to_use = correlations[correlations.abs() > 0.1].index.tolist()
        X = grade_data[features_to_use].values
        y = grade_data['ImpactAvg'].values
        
        predictions = train_and_evaluate(X, y)
        model_metrics[grade]['predictions']['ImpactAvg'] = predictions
        model_metrics[grade]['actual_values']['ImpactAvg'] = y
    
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
    """Check if a value meets the requirements for the given grade and property"""
    if property_type == 'ImpactAvg':
        return True  # Skip requirements check for ImpactAvg
    
    requirements = get_min_requirements(16, grade)
    
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
    
    if property_type == 'ImpactAvg':  # Add ImpactAvg handling
        min_req = 27  # Minimum impact value
        return pred_lower >= min_req
    elif property_type == 'Yield Strength [Mpa]':
        min_req = requirements['yield_strength']['min']
        return pred_lower >= min_req
    elif property_type == 'Tensile Strength [in MPA]':
        min_req = requirements['tensile_strength']['min']
        max_req = requirements['tensile_strength']['max']
        if pred_lower == pred_upper == actual:
            return min_req <= actual <= max_req
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
    """Analyze and plot optimal confidence levels for each property"""
    # Only plot properties with requirements (exclude ImpactAvg)
    plot_targets = ['Yield Strength [Mpa]', 'Tensile Strength [in MPA]', 'Elongation']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Changed to 1x3 for three properties
    axes = axes.flatten()
    
    optimal_cis = {}
    
    for idx, target in enumerate(plot_targets):  # Only loop through properties with requirements
        for grade in ['S355', 'S690']:
            if target in metrics[grade]['false_rates']:
                result = metrics[grade]['false_rates'][target]
                plot_confidence_interval_analysis(
                    result['confidence_intervals'],
                    100 - np.array(result['false_positives']['percentage']),
                    result['false_positives']['percentage'],
                    result['false_negatives']['percentage'],
                    grade, target, axes[idx],
                    None
                )
                
    plt.tight_layout()
    return optimal_cis

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

def kelvin_conversion(temp):
    """Convert Celsius to Kelvin"""
    return temp + 273.15

def is_valid_temperature(temp):
    """Check if temperature is in valid range"""
    valid_temps = [-60, -50, -40, -20, -10, 0]
    try:
        return float(temp) in valid_temps
    except (ValueError, TypeError):
        return False

def process_impact_value(value, column_name):
    """Process impact test values, handling split values, (T)/(L) suffixes, and inequality symbols"""
    try:
        if isinstance(value, str):
            # Remove inequality symbols and get numeric value
            value = value.replace('>', '').replace('<', '').strip()

            # Handle (T) or (L) suffix
            if '(T)' in value or '(L)' in value:
                orientation = 'T' if '(T)' in value else 'L'
                numeric_value = value.replace(f'({orientation})', '').strip()
                try:
                    return {orientation: float(numeric_value.replace(',', '.'))}
                except ValueError:
                    return None

            # Handle multiple '/' case
            if value.count('/') > 1:
                return None
            
            # Handle T/L split values
            if '/' in value:
                t_val, l_val = value.split('/')
                try:
                    return {'T': float(t_val.replace(',', '.')), 
                           'L': float(l_val.replace(',', '.'))}
                except ValueError:
                    return None
            
            # Handle single values
            try:
                return {'single': float(value.replace(',', '.'))}
            except ValueError:
                return None
                
        return {'single': float(value) if value else 0}
    except Exception:
        return None

def expand_impact_rows(df):
    """Expand rows with split impact values into separate T/L rows"""
    new_rows = []
    
    for idx, row in df.iterrows():
        # Process impact values with column names
        v1 = process_impact_value(row['QC_V1'], 'QC_V1')
        v2 = process_impact_value(row['QC_V2'], 'QC_V2')
        v3 = process_impact_value(row['QC_V3'], 'QC_V3')
        
        # Skip if any value is None (invalid) or all values are 0
        if any(v is None for v in [v1, v2, v3]):
            continue
        if all(val.get('single', 0) == 0 for val in [v1, v2, v3]):
            continue
            
        # Check temperature validity
        if not is_valid_temperature(row['QC_temp']):
            continue
            
        # Convert temperature to Kelvin
        temp_k = kelvin_conversion(float(row['QC_temp']))
        
        # Handle orientation cases
        orientations = []
        if '/' in str(row['QC_V1']) or '/' in str(row['QC_V2']) or '/' in str(row['QC_V3']):
            orientations = ['T', 'L']
        elif row['QC_Orientation'] in ['T', 'L']:
            orientations = [row['QC_Orientation']]
        else:
            continue  # Skip invalid orientations
            
        # Create new rows for each orientation
        for orientation in orientations:
            new_row = row.copy()
            new_row['QC_temp'] = temp_k
            new_row['QC_Orientation'] = orientation
            
            # Set impact values based on orientation
            if orientation == 'T':
                new_row['QC_V1'] = v1.get('T', v1.get('single', 0))
                new_row['QC_V2'] = v2.get('T', v2.get('single', 0))
                new_row['QC_V3'] = v3.get('T', v3.get('single', 0))
            else:  # L orientation
                new_row['QC_V1'] = v1.get('L', v1.get('single', 0))
                new_row['QC_V2'] = v2.get('L', v2.get('single', 0))
                new_row['QC_V3'] = v3.get('L', v3.get('single', 0))
            
            # Calculate ImpactAvg
            impact_values = [new_row['QC_V1'], new_row['QC_V2'], new_row['QC_V3']]
            non_zero_values = [v for v in impact_values if v != 0]
            new_row['QC_ImpactAvg'] = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0
            
            new_rows.append(new_row)
    
    return pd.DataFrame(new_rows)

def analyze_impact_correlations(df):
    """Analyze feature correlations for Impact Average"""
    # Ensure only numeric columns are considered
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check if 'QC_ImpactAvg' is in the numeric columns
    if 'QC_ImpactAvg' not in numeric_df.columns:
        print("Warning: 'QC_ImpactAvg' is not a numeric column or is missing.")
        return None
    
    # Calculate correlations
    correlations = numeric_df.corr()['QC_ImpactAvg'].sort_values(ascending=False)
    print(correlations)
    return correlations

def calculate_confidence_interval(predictions, confidence=0.975):
    """Calculate confidence interval for predictions"""
    z_score = norm.ppf(confidence)
    std_dev = np.std(predictions)
    margin_of_error = z_score * std_dev
    return margin_of_error

def analyze_impact_accuracy(predictions, actual_values, error_margins=[5, 10, 15, 20]):
    """
    Analyze prediction accuracy for different error margins
    Returns percentage of predictions within each error margin
    """
    results = {}
    total_samples = len(predictions)
    
    for margin in error_margins:
        within_margin = np.sum(np.abs(predictions - actual_values) <= margin)
        percentage = (within_margin / total_samples) * 100
        results[margin] = percentage
    
    return results

def plot_confidence_intervals(intervals, grade):
    """Create dual-axis bar plot for confidence intervals"""
    confidence_levels = list(intervals.keys())
    margins = [data['margin'] for data in intervals.values()]
    accuracies = [data['accuracy'] for data in intervals.values()]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot margins as bars
    x = np.arange(len(confidence_levels))
    bars = ax1.bar(x, margins, color='lightblue', alpha=0.7)
    ax1.set_xlabel('Confidence Level')
    ax1.set_ylabel('Margin of Error (J)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}J',
                ha='center', va='bottom')
    
    # Create second y-axis for accuracy
    ax2 = ax1.twinx()
    line = ax2.plot(x, accuracies, color='red', marker='o', linewidth=2, label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add accuracy values
    for i, accuracy in enumerate(accuracies):
        ax2.text(i, accuracy, f'{accuracy:.1f}%', 
                ha='center', va='bottom', color='red')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(confidence_levels, rotation=45)
    
    plt.title(f'Confidence Interval Analysis - {grade}')
    plt.tight_layout()
    
    return fig

def main():
    """Compare different confidence intervals for ImpactAvg with plots"""
    metrics = process_data_and_train()
    
    for grade in ['S355', 'S690']:
        predictions = metrics[grade]['predictions']['ImpactAvg']
        actual_values = metrics[grade]['actual_values']['ImpactAvg']
        
        # Calculate confidence intervals
        intervals = calculate_confidence_intervals(predictions, actual_values)
        
        # Create and show plot
        fig = plot_confidence_intervals(intervals, grade)
        plt.show()

def train_and_evaluate(X, y):
    """Helper function to train and evaluate models"""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    predictions = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rt_model = ExtraTreesRegressor(n_estimators=50, random_state=42)
        
        rf_model.fit(X_train_scaled, y_train)
        rt_model.fit(X_train_scaled, y_train)
        
        rf_pred = rf_model.predict(X_test_scaled)
        rt_pred = rt_model.predict(X_test_scaled)
        
        # Use best predictions
        pred = np.where(
            np.abs(rf_pred - y_test) < np.abs(rt_pred - y_test),
            rf_pred,
            rt_pred
        )
        predictions.extend(pred)
    
    return predictions

def compare_encoding_methods(df):
    """Compare original vs cumulative temperature encoding"""
    results = {}
    
    for grade in ['S355', 'S690']:
        grade_data = df[df['Simplified Grade (for further analysis)'] == grade]
        
        # Test both encoding methods
        for method in ['original', 'cumulative']:
            # Create features using appropriate method
            if method == 'original':
                enhanced_data = create_enhanced_features_original(grade_data)
            else:
                enhanced_data = create_enhanced_features(grade_data)
            
            # Prepare data for modeling
            features = [col for col in enhanced_data.columns 
                       if col not in targets + ['Manufacturer', 'Simplified Grade (for further analysis)']
                       and col != 'Grade (as reported)'
                       and enhanced_data[col].dtype != 'object']
            
            X = enhanced_data[features].values
            y = enhanced_data['ImpactAvg'].values
            
            # Perform cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            fold_metrics = []
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train and predict
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                predictions = rf_model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                # Calculate temperature-specific metrics
                temp_metrics = {}
                for temp in [-60, -50, -40, -20, -10, 0]:
                    # Use QC_temp directly instead of QC_temp_C
                    temp_mask = grade_data['QC_temp'].iloc[test_idx] == (temp + 273.15)  # Convert to Kelvin
                    if np.sum(temp_mask) > 0:
                        temp_metrics[temp] = {
                            'mae': mean_absolute_error(y_test[temp_mask], predictions[temp_mask]),
                            'samples': np.sum(temp_mask)
                        }
                
                fold_metrics.append({
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'temp_metrics': temp_metrics
                })
            
            # Aggregate results
            results[f"{grade}_{method}"] = {
                'mse': np.mean([m['mse'] for m in fold_metrics]),
                'mae': np.mean([m['mae'] for m in fold_metrics]),
                'r2': np.mean([m['r2'] for m in fold_metrics]),
                'temp_metrics': {}
            }
            
            # Aggregate temperature-specific metrics
            for temp in [-60, -50, -40, -20, -10, 0]:
                temp_maes = []
                total_samples = 0
                for m in fold_metrics:
                    if temp in m['temp_metrics']:
                        temp_maes.append(m['temp_metrics'][temp]['mae'] * m['temp_metrics'][temp]['samples'])
                        total_samples += m['temp_metrics'][temp]['samples']
                
                if total_samples > 0:
                    results[f"{grade}_{method}"]['temp_metrics'][temp] = {
                        'mae': sum(temp_maes) / total_samples,
                        'samples': total_samples
                    }
    
    return results

def print_comparison_results(results):
    """Print formatted comparison results"""
    print("\nComparison of Temperature Encoding Methods:")
    print("=" * 80)
    
    for grade in ['S355', 'S690']:
        print(f"\n{grade} Results:")
        print("-" * 40)
        
        orig = f"{grade}_original"
        cum = f"{grade}_cumulative"
        
        # Print overall metrics
        print("\nOverall Metrics:")
        print(f"{'Metric':<15} {'Original':<15} {'Cumulative':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for metric in ['mse', 'mae', 'r2']:
            orig_val = results[orig][metric]
            cum_val = results[cum][metric]
            improvement = ((orig_val - cum_val) / orig_val * 100 
                         if metric != 'r2' else 
                         ((cum_val - orig_val) / abs(orig_val)) * 100)
            
            print(f"{metric.upper():<15} {orig_val:>15.3f} {cum_val:>15.3f} {improvement:>14.1f}%")
        
        # Print temperature-specific metrics
        print("\nTemperature-Specific MAE:")
        print(f"{'Temp (°C)':<15} {'Original':<15} {'Cumulative':<15} {'Improvement':<15} {'Samples':<10}")
        print("-" * 70)
        
        for temp in sorted(results[orig]['temp_metrics'].keys()):
            if temp in results[cum]['temp_metrics']:
                orig_mae = results[orig]['temp_metrics'][temp]['mae']
                cum_mae = results[cum]['temp_metrics'][temp]['mae']
                improvement = ((orig_mae - cum_mae) / orig_mae * 100)
                samples = results[orig]['temp_metrics'][temp]['samples']
                
                print(f"{temp:<15} {orig_mae:>15.3f} {cum_mae:>15.3f} {improvement:>14.1f}% {samples:>10}")

if __name__ == "__main__":
    main()