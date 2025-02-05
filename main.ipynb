!pip install pulp
!pip install pandas
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def run_dea_analysis(input_file, output_file):
    """
    Function to perform DEA analysis using Constant Returns to Scale (CRS).
    """
    # 1. Read data from input file
    df = pd.read_csv(input_file)
    # 2. Data preprocessing
    df['Execution Speed'] = 1 / df['Execution Speed/second']  # Convert execution speed to a positive metric
    # 3. Define input-output variables
    input_columns = [
        'Development Time (hours)',
        'Labor Effort (man-hours)',
        'Model Training Time (hours)',
        'Data (GB)',
        'IT Infrastructure (servers)',
        'CPU (cores)',
        'Memory (GB)'
    ]
    output_columns = [
        'Output - Model Robustness (1-10)',
        'Output - Model Performance (%)',
        'Execution Speed'
    ]
    inputs = df[input_columns].values
    outputs = df[output_columns].values
    # 4. DEA Model
    efficiency_scores = []
    for dmu in range(len(df)):
        # Set up optimization model
        prob = pulp.LpProblem(f"DEA_DMU_{dmu}", pulp.LpMinimize)
        theta = pulp.LpVariable('theta', lowBound=0)
        lambdas = pulp.LpVariable.dicts('lambda', range(len(df)), lowBound=0)
        # Objective function
        prob += theta
        # Input constraints
        for i in range(inputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(df))]) <= theta * inputs[dmu][i]
        # Output constraints
        for r in range(outputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(df))]) >= outputs[dmu][r]
        # Solve optimization
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        efficiency_scores.append(round(pulp.value(theta), 3))
    # 5. Save results
    df['Efficiency'] = efficiency_scores
    result_df = df[['DMU', 'Efficiency']].sort_values(by='Efficiency', ascending=False)
    result_df.to_csv(output_file, sep='\t', index=False)
    print("DEA-CRS (Constant Returns to Scale) Analysis Results")
    print(result_df)
    return df

def run_dea_vrs_analysis(input_file, output_file):
    """
    Function to perform DEA analysis using Variable Returns to Scale (VRS).
    """
    # 1. Read data from input file
    df = pd.read_csv(input_file)
    # 2. Data preprocessing
    df['Execution Speed'] = 1 / df['Execution Speed/second']  # Convert execution speed to a positive metric
    print(df['Execution Speed'])

    # 3. Define input-output variables
    input_columns = [
        'Development Time (hours)',
        'Labor Effort (man-hours)',
        'Model Training Time (hours)',
        'Data (GB)',
        'IT Infrastructure (servers)',
        'CPU (cores)',
        'Memory (GB)'
    ]
    output_columns = [
        'Output - Model Robustness (1-10)',
        'Output - Model Performance (%)',
        'Execution Speed'
    ]
    inputs = df[input_columns].values
    outputs = df[output_columns].values
    # 4. DEA-VRS Model
    efficiency_scores = []
    for dmu in range(len(df)):
        prob = pulp.LpProblem(f"DEA_VRS_DMU_{dmu}", pulp.LpMinimize)
        theta = pulp.LpVariable('theta', lowBound=0)
        lambdas = pulp.LpVariable.dicts('lambda', range(len(df)), lowBound=0)
        prob += theta
        for i in range(inputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * inputs[j][i] for j in range(len(df))]) <= theta * inputs[dmu][i]
        for r in range(outputs.shape[1]):
            prob += pulp.lpSum([lambdas[j] * outputs[j][r] for j in range(len(df))]) >= outputs[dmu][r]
        prob += pulp.lpSum([lambdas[j] for j in range(len(df))]) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        efficiency_scores.append(round(pulp.value(theta), 3))
    # 5. Save results
    df['Efficiency'] = efficiency_scores
    result_df = df[['DMU', 'Efficiency']].sort_values(by='Efficiency', ascending=False)
    result_df.to_csv(output_file, sep='\t', index=False)
    print("DEA-VRS (Variable Returns to Scale) Analysis Results")
    print(result_df)
    return df

def plot_correlation_heatmap(df):
    """
    Function to visualize the correlation between input/output variables and efficiency.
    """
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap of Variable Correlations')
    plt.show()

def plot_regression_analysis(df):
    """
    Function to visualize the influence of variables on efficiency using linear regression.
    """
    X = df.drop(columns=['DMU', 'Efficiency'])
    y = df['Efficiency']
    model = LinearRegression()
    model.fit(X, y)
    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Variable', data=coefficients, palette='viridis')
    plt.title('Impact of Variables on Efficiency (Linear Regression)')
    plt.xlabel('Regression Coefficient')
    plt.ylabel('Variable')
    plt.show()

def plot_sensitivity_analysis(df):
    """
    Function to visualize variable sensitivity to efficiency.
    """
    sensitivity_vars = ['Development Time (hours)', 'Model Training Time (hours)','IT Infrastructure (servers)','Labor Effort (man-hours)']
    sensitivity_df = df[sensitivity_vars + ['Efficiency']]
    fig, axes = plt.subplots(1, len(sensitivity_vars), figsize=(18, 5))
    for i, var in enumerate(sensitivity_vars):
        sns.scatterplot(data=sensitivity_df, x=var, y='Efficiency', ax=axes[i])
        axes[i].set_title(f'Sensitivity Analysis: {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Efficiency')
    plt.tight_layout()
    plt.show()

def plot_slack_analysis(df):
    """
    Function to visualize slack (areas needing improvement).
    """
    slack_vars = ['Development Time (hours)', 'Labor Effort (man-hours)', 'Model Training Time (hours)']
    slack_df = df[slack_vars + ['Efficiency']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=slack_df, orient='h', palette='Set2')
    plt.title('Slack Analysis: Areas for Improvement')
    plt.xlabel('Input Value')
    plt.ylabel('Input Variable')
    plt.show()

def plot_efficiency_results(df):
    """
    Function to visualize DEA-VRS efficiency results using a vertical bar chart with vertical efficiency scores on the left.
    """
    # Extract DMU names and efficiency scores
    dmus = df['DMU']
    efficiencies = df['Efficiency']

    # Plot vertical bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(dmus, efficiencies, color=['green' if eff == 1.0 else 'orange' for eff in efficiencies])

    # Add efficiency values on the left side of the bars (rotated vertically)
    for bar, efficiency in zip(bars, efficiencies):
        plt.text(
            bar.get_x() - 0.1,  # Position text slightly to the left of the bar
            bar.get_height() / 2 + bar.get_y(),  # Center vertically
            f'{efficiency:.3f}',
            va='center',
            fontsize=10,
            rotation=90  # Rotate text vertically
        )

    # Title and labels
    plt.title("DEA-VRS (Variable Returns to Scale) Analysis Results", fontsize=16)
    plt.ylabel("Efficiency Score", fontsize=14)
    plt.xlabel("DMU (Decision Making Unit)", fontsize=14)

    # Rotate DMU names for better readability
    plt.xticks(rotation=45, ha='right')

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_crs_efficiency_results(df):
    """
    Function to visualize DEA-VRS efficiency results using a vertical bar chart with vertical efficiency scores on the left.
    """
    # Extract DMU names and efficiency scores
    dmus = df['DMU']
    efficiencies = df['Efficiency']

    # Plot vertical bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(dmus, efficiencies, color=['green' if eff == 1.0 else 'orange' for eff in efficiencies])

    # Add efficiency values on the left side of the bars (rotated vertically)
    for bar, efficiency in zip(bars, efficiencies):
        plt.text(
            bar.get_x() - 0.1,  # Position text slightly to the left of the bar
            bar.get_height() / 2 + bar.get_y(),  # Center vertically
            f'{efficiency:.3f}',
            va='center',
            fontsize=10,
            rotation=90  # Rotate text vertically
        )

    # Title and labels
    plt.title("DEA-CRS (Constant Returns to Scale) Analysis Results", fontsize=16)
    plt.ylabel("Efficiency Score", fontsize=14)
    plt.xlabel("DMU (Decision Making Unit)", fontsize=14)

    # Rotate DMU names for better readability
    plt.xticks(rotation=45, ha='right')

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Run DEA-VRS analysis
    df_vrs = run_dea_vrs_analysis('input.csv', 'output-vrs.txt')
    # Run DEA-CRS analysis
    df_crs = run_dea_analysis('input.csv', 'output-crs.txt')
    # Perform correlation analysis
    plot_correlation_heatmap(df_vrs)
    # Perform sensitivity analysis
    plot_sensitivity_analysis(df_vrs)
    # Perform slack analysis
    plot_slack_analysis(df_vrs)
    # Plot DEA-VRS efficiency results
    plot_efficiency_results(df_vrs)
      # Plot DEA-VRS efficiency results
    plot_crs_efficiency_results(df_crs)
