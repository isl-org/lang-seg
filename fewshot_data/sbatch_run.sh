#!/bin/bash
DATE=`date -d now`
EXP=hsnet
NGPU=4
partition=g24
JOB=fewshot_${EXP}
SAVE_ROOT="save/${EXP}"
SCRIPT_ROOT="sweep_scripts/${EXP}"
mkdir -p $SCRIPT_ROOT
NCPU=$((NGPU * 10))
qos=normal  # high normal low

function print_append {
    echo $@ >> $SCRIPT
}

function slurm_append {
    echo $@ >> $SLURM
}

function print_setup {
    SAVE="${SAVE_ROOT}/${JOB}"
    SCRIPT="${SCRIPT_ROOT}/${JOB}.sh"
    SLURM="${SCRIPT_ROOT}/${JOB}.slrm"
    mkdir -p $SAVE
    echo `date -d now` $SAVE >> 'submitted.txt'
    echo "#!/bin/bash" > $SLURM
    slurm_append "#SBATCH --job-name=job1111_${JOB}"
    slurm_append "#SBATCH --output=${SAVE}/stdout.txt"
    slurm_append "#SBATCH --error=${SAVE}/stderr.txt"
    slurm_append "#SBATCH --open-mode=append"
    slurm_append "#SBATCH --signal=B:USR1@120"

    slurm_append "#SBATCH -p ${partition}"
    slurm_append "#SBATCH --gres=gpu:${NGPU}"
    slurm_append "#SBATCH -c ${NCPU}"
    slurm_append "#SBATCH -t 02-00"
    # slurm_append "#SBATCH -t 01-00"
    # slurm_append "#SBATCH -t 00-06"
    slurm_append "#SBATCH --qos=${qos}" 
    slurm_append "srun sh ${SCRIPT}"

    echo "#!/bin/bash" > $SCRIPT
    print_append "trap_handler () {"
    print_append "echo \"Caught signal: \" \$1"
    print_append "# SIGTERM must be bypassed"
    print_append "if [ "$1" = "TERM" ]; then"
    print_append "echo \"bypass sigterm\""
    print_append "else"
    print_append "# Submit a new job to the queue"
    print_append "echo \"Requeuing \" \$SLURM_JOB_ID"
    print_append "scontrol requeue \$SLURM_JOB_ID"
    print_append "fi"
    print_append "}"
    print_append "trap 'trap_handler USR1' USR1"
    print_append "trap 'trap_handler TERM' TERM"

    print_append "{"
    print_append "source activate pytorch"
    print_append "conda activate pytorch"
    print_append "export PATH=/home/boyili/programfiles/anaconda3/envs/pytorch/bin:/home/boyili/programfiles/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
    print_append "which python"
    print_append "echo \$PATH"
    print_append "export NCCL_DEBUG=INFO"
    print_append "export PYTHONFAULTHANDLER=1"

    echo $JOB
}

function print_after {
    print_append "} & "
    print_append "wait \$!"
    print_append "sleep 610 &"
    print_append "wait \$!"
}

print_setup
print_append stdbuf -o0 -e0 \
    python train.py --log 'log_pascal'
print_after
sbatch $SLURM
