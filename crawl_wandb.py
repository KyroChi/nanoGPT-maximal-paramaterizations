"""
This script crawls data from Weights & Biases (wandb) for all running or finished
experiments within a specified project.

It is typically invoked by `run_crawl_wandb.sh`, which provides the project name,
entity, and output directory via command-line arguments.

It performs the following steps:
1.  Parses command-line arguments for project name, entity, and output directory.
2.  Initializes the wandb API.
3.  Fetches *all* runs from the specified project and entity that are currently in
    either the 'running' or 'finished' state.
4.  For each fetched run, it extracts the history for hardcoded metric names
    (`lm loss validation`, `lm loss`), along with the run's config and summary.
5.  Saves the collected data in two ways into the specified output directory:
    a.  **Consolidated:** Combines history from all runs into `all_data.csv`,
        all configurations into `all_configs.json`, and all summaries into
        `all_summaries.json`.
    b.  **Individual:** For each run, creates a subdirectory named after the run
        (sanitized) and saves its individual `data.csv`, `config.json`,
        `summary.json`, and `metadata.json`.
"""
import wandb
import pandas as pd
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import itertools


def crawl_wandb_data(project_name, entity, metric_names, run_name_filter=None):
    """
    Crawl wandb data for specific runs and metrics.
    
    Args:
        project_name (str): Name of the wandb project
        metric_names (str or list): Name(s) of the metric(s) to extract
        entity (str, optional): The entity (username or team name). Defaults to None.
        run_name_filter (str, optional): Filter to apply to run names. Defaults to None.
    
    Returns:
        dict: Dictionary of run_name -> DataFrame with metric data
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Convert single metric to list for consistent handling
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    # Get all runs from the project that are in running state
    # Note: We're not filtering by run_name_filter to keep all running experiments
    project_path = f"{entity}/{project_name}" if entity else project_name
    # runs = api.runs(project_path, filters={"state": ["crashed", "finished"]})
    runs_finished = api.runs(project_path, filters={"state": "finished"})
    runs_crashed = api.runs(project_path, filters={"state": "crashed"})

    # runs = itertools.chain(runs_finished, runs_crashed)
    runs = runs_finished

    
    print(f"Found {len(runs_finished)} finished and {len(runs_crashed)} crashed / unfinished experiments")
    
    # Extract data for each run
    run_data = {}
    for run in (pbar := tqdm(runs)):
        try:
            # Get history of the run with all requested metrics
            history = run.history(keys=metric_names, pandas=True)
            
            # Check if at least one of the metrics is in the history
            if any(metric in history.columns for metric in metric_names):
                # Add run ID to identify the source in combined data
                history['run_id'] = run.id
                history['run_created_at'] = run.created_at
                history['run_name'] = run.name
                
                run_data[run.id] = {
                    'data': history,
                    'config': run.config,
                    'summary': run.summary._json_dict,
                    'created_at': run.created_at,
                    'name': run.name
                }
            else:
                print(f"Warning: None of the metrics {metric_names} found in run {run.name}")
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
    
    return run_data

def save_data(run_data, output_dir="wandb_data"):
    print(f"Saving data to {output_dir}...")
    print(f"Total runs processed: {len(run_data)}")
    """Save the extracted data to consolidated files."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not run_data:
        print("No data to save.")
        return
    
    # Concatenate all history data
    all_data = []
    run_configs = {}
    run_summaries = {}
    
    for run_id, data in run_data.items():
        all_data.append(data['data'])
        run_configs[run_id] = data['config']
        run_summaries[run_id] = data['summary']
    
    if all_data:
        # Combine all data and save to one CSV
        all_data_df = pd.concat(all_data, ignore_index=True)
        all_data_df.to_csv(f"{output_dir}/all_data.csv")
        
        # Save all configs and summaries
        with open(f"{output_dir}/all_configs.json", 'w') as f:
            json.dump(run_configs, f, indent=2)
        
        with open(f"{output_dir}/all_summaries.json", 'w') as f:
            json.dump(run_summaries, f, indent=2)
        
        # Also save individual run data
        for run_id, data in run_data.items():
            # Use run name for the directory name
            run_name = data['name']
            # Replace any characters that might be invalid in directory names
            safe_run_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in run_name)
            run_dir = os.path.join(output_dir, safe_run_name)
            os.makedirs(run_dir, exist_ok=True)
            
            # Save individual run data
            data['data'].to_csv(f"{run_dir}/data.csv")
            
            with open(f"{run_dir}/config.json", 'w') as f:
                json.dump(data['config'], f, indent=2)
            
            with open(f"{run_dir}/summary.json", 'w') as f:
                json.dump(data['summary'], f, indent=2)
            
            with open(f"{run_dir}/metadata.json", 'w') as f:
                json.dump({
                    "run_id": run_id,
                    "name": data['name'],
                    "created_at": str(data['created_at'])
                }, f, indent=2)
        
        print(f"Data saved to {output_dir}:")
        print(f"- All data: all_data.csv")
        print(f"- All configs: all_configs.json")
        print(f"- All summaries: all_summaries.json")
        print(f"- Individual run data saved in separate directories by run name")


def parse_args():
    parser = argparse.ArgumentParser(description='Crawl and process W&B data for running experiments.')
    parser.add_argument('--project', type=str, required=True, help='Name of the wandb project')
    parser.add_argument('--entity', type=str, default=None, help='The entity (username or team name)')
    parser.add_argument('--output-dir', type=str, default='wandb_data', help='Directory to save output files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    metrics = ['train/loss', 'val/loss'] #["train/loss", "train/avg_loss", "tiwd", "val/loss"]
    # Crawl data using command line arguments - no run_name_filter to get all running experiments
    run_data = crawl_wandb_data(
        project_name=args.project, 
        entity=args.entity, 
        metric_names=metrics,
        run_name_filter=None
    )
    
    # Save combined data to consolidated files
    save_data(run_data, args.output_dir)

    print("Data crawling and processing completed!")