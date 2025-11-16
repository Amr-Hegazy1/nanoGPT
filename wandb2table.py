import wandb
import pandas as pd

# Configure your W&B details here
ENTITY = "amr-hegazy-german-university-in-cairo"  # Your W&B username or team name
PROJECT = "nanoGPT-experiments-gpt2"  # Your W&B project name

def get_experiments_table():
    """
    Fetch all experiments from a W&B project and create a table
    with experiment names and their final validation loss.
    """
    # Initialize W&B API
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    # Prepare data for the table
    data = []
    
    for run in runs:
        # Get run name
        name = run.name
        
        # Try to get the final validation loss
        # Check common metric names for validation loss
        val_loss = None
        possible_names = ['val/loss', 'val_loss', 'validation_loss', 'validation/loss']
        
        for metric_name in possible_names:
            if metric_name in run.summary:
                val_loss = run.summary[metric_name]
                break
        
        # If not found in summary, try history (last value)
        if val_loss is None:
            history = run.history()
            if not history.empty:
                for metric_name in possible_names:
                    if metric_name in history.columns:
                        val_loss = history[metric_name].dropna().iloc[19] if not history[metric_name].dropna().empty else None
                        break
        
        data.append({
            'Experiment Name': name,
            'Final Val Loss': val_loss if val_loss is not None else 'N/A',
            'State': run.state,
            'Created': run.created_at
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by experiment name
    df = df.sort_values('Experiment Name')
    
    return df

if __name__ == "__main__":
    # Make sure you're logged in to W&B
    # Run: wandb login
    
    print(f"Fetching experiments from {ENTITY}/{PROJECT}...")
    
    try:
        df = get_experiments_table()
        
        print(f"\nFound {len(df)} experiments:\n")
        print(df.to_string(index=False))
        
        # Optionally save to CSV
        df.to_csv('wandb_experiments.csv', index=False)
        print("\nTable saved to wandb_experiments.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you:")
        print("1. Have run 'wandb login' to authenticate")
        print("2. Have set the correct ENTITY and PROJECT names")
        print("3. Have access to the specified project")