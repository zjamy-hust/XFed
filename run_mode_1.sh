if [ ! -d "res" ]; then
  mkdir res
fi

mode=1
# python -u fedxai.py \
python -u fedxai.py \
            --epochs 20 \
            --num_users 16 \
            --client_selection_num 16 \
            --local_ep 50 \
            --local_bs 128 \
            --lr 0.1 \
            --model resnet18 \
            --save_global 1 \
            --dataset 'cifar10'\
            --iid 1 \
            --compare_sever_client_masks 1\
            --XAI_evaluate_batch_size 16\
            --XAI_evaluate_nt_samples 5\
            --XAI_evaluate_n_steps 5\
            --mode $mode \
            --test_mask_batch_size 20 \
            --test_mask_nt_samples 10 \
            --test_mask_n_steps 15 \
            --gpu 1 #\
            # > ./res/mode_${mode}.log 2>&1 &