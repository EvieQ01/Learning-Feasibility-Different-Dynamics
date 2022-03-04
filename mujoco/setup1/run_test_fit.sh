method=main_feasibility
xml=1.9
# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1 \
#         --demo_files ../demo/walker2d_19.9/batch_00.pkl \
#         --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
#         --xml ${xml}  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.1 --mode traj --discount_train
python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1_fit_test_${xml} \
        --xml walker2d_${xml}.xml  \
        --demo_files ../demo/walker2d_19.9/batch_00.pkl \
        --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
        --env-name CustomWalker2dFeasibility-v0 \
        --ratio 0.1  --discount_train --mode test


        
