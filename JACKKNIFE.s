


#PBS -N JK
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00 
#PBS -q small
#PBS -o /home/chenzy/code/SZ_planck_DESI/pairwise_ksz/eolog/jackknife
#PBS -e /home/chenzy/code/SZ_planck_DESI/pairwise_ksz/eolog/jackknife1

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
module load miniconda/miniconda3
conda activate py33

cd /home/chenzy/code/SZ_planck_DESI/pairwise_kSZ_paper

python Operate_baseline.py

wait

echo over


