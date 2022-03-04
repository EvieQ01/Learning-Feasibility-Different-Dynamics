method=main_feasibility
xml=0.7
n_domains=3
mode=resplit-far
# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1 \
#         --demo_files ../demo/walker2d_19.9/batch_00.pkl \
#         --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
#         --xml ${xml}  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.1 --mode traj --discount_train
python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1_resplit_${xml}_${n_domains}_${mode} \
        --xml walker2d_${xml}.xml  \
        --env-name CustomWalker2dFeasibility-v0 \
        --ratio 1. --mode ${mode} --discount_train --n_domains ${n_domains}

# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1_resplit_${xml}_${n_domains} \
#         --xml walker2d_${xml}.xml  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.05 --mode random-split --discount_train --n_domains ${n_domains}

        
