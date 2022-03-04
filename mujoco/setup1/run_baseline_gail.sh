cuda=3
method=main_gailfo
xml=0.7
n_domains=2
mode=baseline_notsim
export CUDA_VISIBLE_DEVICES=${cuda}
# python ${method}.py --save_path checkpoints/walkerπ2d_feasibility_setup1 \
#         --demo_files ../demo/walker2d_19.9/batch_00.pkl \
#         --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
#         --xml ${xml}  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.1 --mode traj --discount_train
python ${method}.py --save_path checkpoints/walker2d_setup1_resplit_${xml}_${n_domains}_${mode} \
        --xml walker2d_${xml}.xml  \
        --demo_files \
                ../demo/walker2d_19.9/batch_00.pkl \
                ../demo/walker2d_29.9/batch_00.pkl \
        --env-name CustomWalker2d-v0 \
        --ratio 0.06 --mode ${mode} --n_domains ${n_domains} \
        --eval-interval 5 --num-epochs 20000
# python main_gailfo.py --env-name CustomWalker2d-v0 --demo_files ../demo/walker2d_19.9/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl --save_path checkpoints/walker2d_imitate1.pth \
#                 --eval-interval 5 --num-epochs 20000 --ratios 0.1 0.1 1 1 --xml walker2d_24.9.xml \
#                 --feasibility_model checkpoints/walker2d_feasibility_setup1 --mode traj
#                 ../demo/walker2d_1.1/batch_00.pkl \
#                 ../demo/walker2d/batch_00.pkl \
                # ../demo/walker2d_9.9/batch_00.pkl \
                # ../demo/walker2d_19.9/batch_00.pkl \
                # ../demo/walker2d_29.9/batch_00.pkl \
                # ../demo/walker2d_27.9/batch_00.pkl \
      
