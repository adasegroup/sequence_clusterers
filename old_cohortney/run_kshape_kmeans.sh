# bash run_kshape_kmeans.sh ${data_path}
#!/usr/bin/env bash
DATA_PATH=${1:-"../pp_clustering/data"} 
mkdir experiments1
for dataset in 'K4_C5' 'trunc_K4_C5' 'K3_C5' 'trunc_K3_C5' 'sin_K5_C5' 'sin_K3_C5' 'IPTV' 'K5_C5' 'ATM' 'sin_K4_C5' 'sin_K2_C5' 'Linkedin' 'trunc_K5_C5' 'trunc_K2_C5' 'Age'; do
   echo Working with ${dataset} 
   python Kshape.py --data_dir ${DATA_PATH}/${dataset} --save_dir experiments1/${dataset}
done


