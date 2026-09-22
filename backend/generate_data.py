import pandas as pd
import numpy as np
import os

def generate_mock_eco_data(num_samples=1000, output_path="backend/historical_eco_data.csv"):
    np.random.seed(42)
    
    # Input Features
    complexity_score = np.random.uniform(1.0, 10.0, num_samples)
    department_count = np.random.randint(1, 6, num_samples)
    original_cost_usd = np.random.uniform(1000, 500000, num_samples)
    
    domains = ["Mechanical", "Electrical", "Software"]
    change_domain = np.random.choice(domains, num_samples)
    
    # Target Variables and Correlations
    
    # Base risk score calculation
    # Higher complexity and more departments involved increase risk
    base_risk = (complexity_score / 10.0) * 0.6 + (department_count / 5.0) * 0.4
    
    # Add some domain specific bias (e.g., Software might have slightly higher unpredictable risk)
    domain_risk_modifier = np.where(change_domain == "Software", 0.1, 0)
    domain_risk_modifier = np.where(change_domain == "Electrical", 0.05, domain_risk_modifier)
    
    total_risk_score = base_risk + domain_risk_modifier + np.random.normal(0, 0.1, num_samples)
    
    # Categorize Risk Level
    risk_level = []
    for score in total_risk_score:
        if score < 0.4:
            risk_level.append("Low")
        elif score < 0.7:
            risk_level.append("Medium")
        else:
            risk_level.append("High")
            
    # Cost Overrun Calculation
    # High risk correlates with higher cost overruns
    # Complexity also directly impacts cost overrun percentage
    cost_overrun_pct = (complexity_score / 10.0) * 0.15 + (department_count / 5.0) * 0.10 + np.random.normal(0.05, 0.05, num_samples)
    cost_overrun_pct = np.clip(cost_overrun_pct, -0.05, 0.5) # Allow some small savings, max 50% overrun
    
    # Additional penalty for High Risk
    risk_penalty = np.where(np.array(risk_level) == "High", 0.1, 0)
    cost_overrun_pct += risk_penalty
    
    cost_overrun_usd = original_cost_usd * cost_overrun_pct
    
    # Create DataFrame
    df = pd.DataFrame({
        "complexity_score": complexity_score,
        "department_count": department_count,
        "original_cost_usd": original_cost_usd,
        "change_domain": change_domain,
        "risk_level": risk_level,
        "cost_overrun_usd": cost_overrun_usd
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} ECO records at {output_path}")

if __name__ == "__main__":
    generate_mock_eco_data()
