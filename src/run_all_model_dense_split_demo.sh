#! /bin/sh
#
# run_all_model_dense_split_demo.sh


for model in H2GCN #FAGCN APPNP GCN GAT ChebNet APPNP JKNet GPRGNN
do
    python train_model.py --RPMAX 10 \
        --net $model \
        --train_rate 0.6 \
        --val_rate 0.2 \
        --dataset squirrel \
        --lr 0.05 \
        --alpha 1.0
done
