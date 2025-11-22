#!/bin/bash
# Quick job monitoring script
# Usage: ./monitor_job.sh [JOB_ID]
# If no JOB_ID provided, uses the most recent job

if [ -z "$1" ]; then
    # Get the most recent job ID
    JOB_ID=$(squeue --me --format='%.10i' --noheader | head -1)
    if [ -z "$JOB_ID" ]; then
        echo "No running jobs found. Please provide a job ID:"
        echo "  ./monitor_job.sh <JOB_ID>"
        exit 1
    fi
    echo "Using most recent job: $JOB_ID"
else
    JOB_ID=$1
fi

echo "Monitoring job $JOB_ID"
echo "Press Ctrl+C to stop"
echo "===================="

while squeue -j $JOB_ID &>/dev/null; do
    clear
    echo "Job Status: $(date)"
    echo "===================="
    squeue -j $JOB_ID --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
    echo ""
    
    if [ -f "grpo_${JOB_ID}.out" ]; then
        echo "Last 15 lines of output:"
        echo "------------------------"
        tail -n 15 grpo_${JOB_ID}.out
    else
        echo "Output file not found yet..."
    fi
    
    echo ""
    if [ -f "grpo_${JOB_ID}.err" ]; then
        ERR_LINES=$(wc -l < grpo_${JOB_ID}.err)
        if [ "$ERR_LINES" -gt 0 ]; then
            echo "Last 10 lines of errors:"
            echo "------------------------"
            tail -n 10 grpo_${JOB_ID}.err
        else
            echo "No errors so far ✓"
        fi
    else
        echo "Error file not found yet..."
    fi
    
    echo ""
    echo "Refresh in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done

echo ""
echo "Job $JOB_ID has completed!"
echo ""
echo "Final output:"
echo "============="
if [ -f "grpo_${JOB_ID}.out" ]; then
    tail -n 30 grpo_${JOB_ID}.out
fi

