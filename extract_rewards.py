#!/usr/bin/env python3
"""
Extract all rewards from a GRPO output file and save every 4th reward to JSON.
"""

import re
import json
from pathlib import Path


def extract_rewards_from_file(file_path):
    """
    Extract all rewards from the output file.
    Returns a list of all reward values.
    """
    all_rewards = []
    
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
                    all_rewards.extend(rewards)
    
    return all_rewards


def main():
    # Input file
    input_file = Path('outputs/grpo_8743727.out')
    
    # Extract all rewards
    print(f"Extracting rewards from {input_file}...")
    all_rewards = extract_rewards_from_file(input_file)
    print(f"Found {len(all_rewards)} total rewards")
    
    # Extract every 4th reward (1-indexed: 1st, 5th, 9th, etc. = indices 0, 4, 8, ...)
    every_4th_reward = all_rewards[::4]
    print(f"Extracted {len(every_4th_reward)} rewards (every 4th)")
    
    # Save to JSON
    output_file = Path('outputs/grpo_8743727_every_4th_reward.json')
    with open(output_file, 'w') as f:
        json.dump(every_4th_reward, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    # Print some stats
    if every_4th_reward:
        print(f"\nStatistics:")
        print(f"  Min: {min(every_4th_reward):.6f}")
        print(f"  Max: {max(every_4th_reward):.6f}")
        print(f"  Mean: {sum(every_4th_reward) / len(every_4th_reward):.6f}")
        print(f"  First 5: {every_4th_reward[:5]}")
        print(f"  Last 5: {every_4th_reward[-5:]}")


if __name__ == '__main__':
    main()

