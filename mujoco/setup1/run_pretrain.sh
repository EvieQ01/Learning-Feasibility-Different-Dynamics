method=main_2phase
# method=re_split_demo
xml=walker2d_24.9.xml

# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1 \
#         --demo_files ../demo/walker2d_19.9/batch_00.pkl \
#         --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
#         --xml ${xml}  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.1 --mode traj --discount_train
python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1 \
        --demo_files ../demo/walker2d_1.1/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl \
        --test_demo_files ../demo/walker2d_1.1/batch_00.pkl ../demo/walker2d_9.9/batch_00.pkl ../demo/walker2d/batch_00.pkl ../demo/walker2d_0.7/batch_00.pkl \
        --xml ${xml}  \
        --env-name CustomWalker2dFeasibility-v0 \
        --ratio 0.1 --mode traj --discount_train
