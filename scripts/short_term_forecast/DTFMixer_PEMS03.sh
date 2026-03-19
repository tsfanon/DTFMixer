model_name=DTFMixer
data_path=PEMS03.npz

seq_len=96
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.003
d_model=128
d_ff=256
batch_size=16
train_epochs=20
patience=10

num_nodes=$(python - <<'PY'
import numpy as np
data = np.load('./dataset/PEMS/PEMS03.npz', allow_pickle=True)['data']
print(data.shape[1])
PY
)

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path $data_path \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len 12 \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in $num_nodes \
 --dec_in $num_nodes \
 --c_out $num_nodes \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window



python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path $data_path \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len 24 \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in $num_nodes \
 --dec_in $num_nodes \
 --c_out $num_nodes \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window



python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path $data_path \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len 48 \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in $num_nodes \
 --dec_in $num_nodes \
 --c_out $num_nodes \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window
