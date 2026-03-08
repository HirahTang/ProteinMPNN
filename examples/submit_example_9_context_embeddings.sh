#!/bin/bash
#SBATCH --job-name=ProteinMPNN_embed
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/ProteinMPNN_logs/embed_job_%A_%a.log
#SBATCH -e /scratch/project_465002574/ProteinMPNN_logs/embed_job_%A_%a.log

set -euo pipefail

echo "============================================================================"
echo "ProteinMPNN Bulk Context Embeddings (Array)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

rocm-smi || echo "Warning: rocm-smi not available"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py_script="/scratch/project_465002574/ProteinMPNN/extract_context_embeddings.py"
if [[ ! -f "$py_script" ]]; then
    echo "Python script not found: $py_script"
    exit 1
fi
echo "Using Python script: $py_script"

input_dir="/scratch/project_465002574/PDB/PDB_cleaned"
output_128_dir="$input_dir/PDB_128"
output_20_dir="$input_dir/PDB_20"
mkdir -p "$output_128_dir" "$output_20_dir"

mapfile -t pdb_files < <(find "$input_dir" -maxdepth 1 -type f -name "*.pdb" | sort)
num_files=${#pdb_files[@]}

if [[ $num_files -eq 0 ]]; then
    echo "No PDB files found in: $input_dir"
    exit 1
fi

num_tasks=${SLURM_ARRAY_TASK_COUNT:-20}
task_id=${SLURM_ARRAY_TASK_ID:-0}
chunk_size=$(( (num_files + num_tasks - 1) / num_tasks ))
start_idx=$(( task_id * chunk_size ))
end_idx=$(( start_idx + chunk_size - 1 ))

if (( start_idx >= num_files )); then
    echo "Task $task_id has no assigned files (num_files=$num_files, num_tasks=$num_tasks)."
    exit 0
fi

if (( end_idx >= num_files )); then
    end_idx=$(( num_files - 1 ))
fi

echo "Total PDB files: $num_files"
echo "Array tasks: $num_tasks"
echo "Task $task_id processing indices: $start_idx..$end_idx"

ok_count=0
fail_count=0

for ((i=start_idx; i<=end_idx; i++)); do
    pdb_path="${pdb_files[$i]}"
    pdb_name="$(basename "$pdb_path" .pdb)"

    out_128="$output_128_dir/${pdb_name}.npz"
    out_20="$output_20_dir/${pdb_name}.npz"

    echo "[$((i+1))/$num_files] Processing: $pdb_name"

    if python "$py_script" \
        --pdb_path "$pdb_path" \
        --model_name v_48_020 \
        --out_file "$out_128" \
        --seed 37 \
        --output_type encoder
    then
        echo "  - Saved 128-d output to: $out_128"
    else
        echo "  - Failed 128-d extraction for $pdb_name"
        fail_count=$((fail_count + 1))
        continue
    fi

    if python "$py_script" \
        --pdb_path "$pdb_path" \
        --model_name v_48_020 \
        --out_file "$out_20" \
        --seed 37 \
        --output_type aa_probs_20
    then
        echo "  - Saved 20-d output to: $out_20"
        ok_count=$((ok_count + 1))
    else
        echo "  - Failed 20-d extraction for $pdb_name"
        fail_count=$((fail_count + 1))
    fi
done

status=0
if (( fail_count > 0 )); then
    status=1
fi

echo "End time: $(date)"
echo "Task summary: success=$ok_count, failed=$fail_count"
if [ $status -eq 0 ]; then
    echo "Embedding extraction finished successfully for all assigned PDBs."
else
    echo "Embedding extraction completed with failures."
fi
exit $status
