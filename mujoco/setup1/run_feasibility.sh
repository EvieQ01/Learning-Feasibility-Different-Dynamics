method=main_feasibility
xml=0.7
n_domains=6
mode=origin
# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1 \
#         --demo_files ../demo/walker2d_19.9/batch_00.pkl \
#         --test_demo_files ../demo/walker2d_19.9/batch_00.pkl  \
#         --xml ${xml}  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.1 --mode traj --discount_train
python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1t_${xml}_${n_domains}}_${mode} \
        --xml walker2d_${xml}.xml  \
        --env-name CustomWalker2dFeasibility-v0 \
        --demo_files ../demo/walker2d_1.1/batch_00.pkl \
                ../demo/walker2d/batch_00.pkl \
                ../demo/walker2d_9.9/batch_00.pkl \
                ../demo/walker2d_19.9/batch_00.pkl \
                ../demo/walker2d_29.9/batch_00.pkl \
                ../demo/walker2d_27.9/batch_00.pkl \
        --test_demo_files ../demo/walker2d_1.1/batch_00.pkl \
                ../demo/walker2d/batch_00.pkl \
                ../demo/walker2d_9.9/batch_00.pkl \
                ../demo/walker2d_19.9/batch_00.pkl \
                ../demo/walker2d_29.9/batch_00.pkl \
                ../demo/walker2d_27.9/batch_00.pkl \
        --ratio .06 --discount_train --n_domains ${n_domains} --mode ${mode}

# python ${method}.py --save_path checkpoints/walker2d_feasibility_setup1_resplit_${xml}_${n_domains} \
#         --xml walker2d_${xml}.xml  \
#         --env-name CustomWalker2dFeasibility-v0 \
#         --ratio 0.05 --mode random-split --discount_train --n_domains ${n_domains}

        
