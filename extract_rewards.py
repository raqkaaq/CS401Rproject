#!/usr/bin/env python3
"""
Extract all reward arrays from a GRPO output file and save every 4th entire array to JSON.
"""

import re
import json
from pathlib import Path


def extract_reward_arrays_from_file(file_path):
    """
    Extract all reward arrays from the output file.
    Returns a list of arrays (each array is a list of reward values).
    """
    all_reward_arrays = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Match lines that start with "Rewards:" (case-insensitive)
            if re.match(r'^Rewards:\s*\[', line, re.IGNORECASE):
                # Extract the list portion
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    # Parse the list of numbers
                    rewards_str = match.group(1)
                    # Split by comma and convert to floats
                    rewards = [float(x.strip()) for x in rewards_str.split(',') if x.strip()]
                    all_reward_arrays.append(rewards)
    
    return all_reward_arrays


def main():
    # Input file
    input_file = Path('outputs/grpo_8766796.out')
    
    # Extract all reward arrays
    print(f"Extracting reward arrays from {input_file}...")
    all_reward_arrays = extract_reward_arrays_from_file(input_file)
    print(f"Found {len(all_reward_arrays)} total reward arrays")
    
    # Extract every 4th array (1-indexed: 1st, 5th, 9th, etc. = indices 0, 4, 8, ...)
    every_4th_array = all_reward_arrays
    print(f"Extracted {len(every_4th_array)} arrays (every 4th array)")
    
    # Calculate total rewards in the extracted arrays
    total_rewards = sum(len(arr) for arr in every_4th_array)
    print(f"Total rewards in extracted arrays: {total_rewards}")
    
    # Save to JSON
    output_file = Path('outputs/middle_new_math_rewards.json')
    with open(output_file, 'w') as f:
        json.dump(every_4th_array, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    # Print some stats
    if every_4th_array:
        # Flatten for statistics
        all_rewards_flat = [reward for arr in every_4th_array for reward in arr]
        print(f"\nStatistics (across all rewards in extracted arrays):")
        print(f"  Total arrays: {len(every_4th_array)}")
        print(f"  Total rewards: {len(all_rewards_flat)}")
        print(f"  Min: {min(all_rewards_flat):.6f}")
        print(f"  Max: {max(all_rewards_flat):.6f}")
        print(f"  Mean: {sum(all_rewards_flat) / len(all_rewards_flat):.6f}")
        print(f"\nFirst array (length {len(every_4th_array[0])}): {every_4th_array[0][:5]}...")
        print(f"Last array (length {len(every_4th_array[-1])}): {every_4th_array[-1][:5]}...")


if __name__ == '__main__':
    main()

