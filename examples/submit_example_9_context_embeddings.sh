#!/bin/bash
#SBATCH --job-name=ProteinMPNN_embed
#SBATCH --account=project_465002574
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o /scratch/project_465002574/ProteinMPNN_logs/embed_job_%A.log
#SBATCH -e /scratch/project_465002574/ProteinMPNN_logs/embed_job_%A.log

echo "============================================================================"
echo "ProteinMPNN Context-Only Position Embeddings"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "============================================================================"

# Load modules for LUMI
module load LUMI
module load CrayEnv
module load lumi-container-wrapper/0.4.2-cray-python-default
export PATH="/flash/project_465002574/unaagi_env/bin:$PATH"

rocm-smi || echo "Warning: rocm-smi not available"

path_to_PDB="/scratch/project_465002574/PDB/PDB_cleaned/7H8V.pdb"
chains_to_use="A B C D"

output_dir="/scratch/project_465002574/ProteinMPNN_outputs/context_embeddings"
mkdir -p "$output_dir"

python ../extract_context_embeddings.py \
    --pdb_path "$path_to_PDB" \
    --pdb_path_chains "$chains_to_use" \
    --model_name v_48_020 \
    --out_file "$output_dir/7H8V_context_embeddings.npz" \
    --seed 37 \
    --output_type aa_log_probs_20

status=$?
echo "End time: $(date)"
if [ $status -eq 0 ]; then
    echo "Embedding extraction finished successfully."
else
    echo "Embedding extraction failed with exit code $status"
fi
exit $status
