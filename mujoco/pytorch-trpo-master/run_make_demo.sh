
# for xml in 0.5 0.7
#     do
#     python main.py --env-name "CustomWalker2d-v0" \
#     --xml walker2d_${xml}.xml
#     done

for xml in 0.5 0.7 #3.9
    do
    python save_traj_trpo.py --env-name "CustomWalker2d-v0" --xml walker2d_${xml}.xml --dump
    done
# for xml in 24.9 24.8 #27.9
#     do
#     python save_traj_trpo.py --env-name "CustomWalker2d-v0" --xml walker2d_${xml}.xml --dump
#     done