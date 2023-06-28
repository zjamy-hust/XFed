mode=2

python -u fedxai.py --epochs 20 --num_users 16 \
                    --local_ep 50 --gpu 1 --iid 1 --mode $mode \
                    --dataset 'cifar10' --num_classes 10 \
                    --test_mask_batch_size 20 \
                    --test_mask_nt_samples 10 --test_mask_n_steps 15 \
                    > ./res/mode_${mode}.log 2>&1