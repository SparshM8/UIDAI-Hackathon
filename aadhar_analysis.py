import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# File paths
files = [
    'api_data_aadhar_enrolment_0_500000.csv',
    'api_data_aadhar_enrolment_500000_1000000.csv',
    'api_data_aadhar_enrolment_1000000_1006029.csv'
]

# Read and combine all CSV files
dfs = []
for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    else:
        print(f"File {file} not found")

if dfs:
    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(data)}")
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
    
    # Basic info
    print(data.info())
    print(data.describe())
    
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())
    
    # Analysis starts here
    
    # 1. Temporal analysis - enrolment trends over time
    daily_enrolment = data.groupby('date').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum', 
        'age_18_greater': 'sum'
    }).reset_index()
    
    daily_enrolment['total'] = daily_enrolment[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
    
    # Plot daily enrolment trends
    plt.figure(figsize=(15, 8))
    plt.plot(daily_enrolment['date'], daily_enrolment['total'], label='Total Enrolments', linewidth=2)
    plt.plot(daily_enrolment['date'], daily_enrolment['age_0_5'], label='Age 0-5', alpha=0.7)
    plt.plot(daily_enrolment['date'], daily_enrolment['age_5_17'], label='Age 5-17', alpha=0.7)
    plt.plot(daily_enrolment['date'], daily_enrolment['age_18_greater'], label='Age 18+', alpha=0.7)
    plt.title('Daily Aadhaar Enrolment Trends by Age Group')
    plt.xlabel('Date')
    plt.ylabel('Number of Enrolments')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('daily_enrolment_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. State-wise analysis
    state_enrolment = data.groupby('state').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    state_enrolment['total'] = state_enrolment[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
    state_enrolment = state_enrolment.sort_values('total', ascending=False)
    
    # Top 10 states
    top_states = state_enrolment.head(10)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(top_states['state'], top_states['total'])
    plt.title('Top 10 States by Total Aadhaar Enrolments')
    plt.xlabel('State')
    plt.ylabel('Total Enrolments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_states_enrolment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Age group distribution
    age_totals = data[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
    
    plt.figure(figsize=(8, 8))
    plt.pie(age_totals, labels=['Age 0-5', 'Age 5-17', 'Age 18+'], autopct='%1.1f%%', startangle=90)
    plt.title('Age Group Distribution in Aadhaar Enrolments')
    plt.axis('equal')
    plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. District-level analysis for top states
    top_state = state_enrolment.iloc[0]['state']
    district_data = data[data['state'] == top_state].groupby('district').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    district_data['total'] = district_data[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
    district_data = district_data.sort_values('total', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(district_data['district'], district_data['total'])
    plt.title(f'Top 10 Districts in {top_state} by Aadhaar Enrolments')
    plt.xlabel('Total Enrolments')
    plt.ylabel('District')
    plt.tight_layout()
    plt.savefig('top_districts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Time series analysis - weekly patterns
    data['week'] = data['date'].dt.isocalendar().week
    data['year'] = data['date'].dt.year
    
    weekly_enrolment = data.groupby(['year', 'week']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    weekly_enrolment['total'] = weekly_enrolment[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(weekly_enrolment)), weekly_enrolment['total'], label='Weekly Total')
    plt.title('Weekly Aadhaar Enrolment Trends')
    plt.xlabel('Week Number')
    plt.ylabel('Number of Enrolments')
    plt.legend()
    plt.tight_layout()
    plt.savefig('weekly_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Correlation analysis
    correlation_matrix = daily_enrolment[['age_0_5', 'age_5_17', 'age_18_greater']].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Between Age Group Enrolments')
    plt.savefig('age_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("\n=== KEY INSIGHTS ===")
    print(f"Total enrolments: {data[['age_0_5', 'age_5_17', 'age_18_greater']].sum().sum():,}")
    print(f"Top state: {top_state} with {state_enrolment.iloc[0]['total']:,} enrolments")
    print(f"Age distribution: {age_totals['age_0_5']/age_totals.sum()*100:.1f}% (0-5), {age_totals['age_5_17']/age_totals.sum()*100:.1f}% (5-17), {age_totals['age_18_greater']/age_totals.sum()*100:.1f}% (18+)")
    
    # Peak days
    peak_day = daily_enrolment.loc[daily_enrolment['total'].idxmax()]
    print(f"Peak enrolment day: {peak_day['date'].strftime('%Y-%m-%d')} with {peak_day['total']:,} enrolments")
    
else:
    print("No data files found")