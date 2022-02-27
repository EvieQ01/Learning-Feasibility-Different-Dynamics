for xml in 0.7 0.5 3.9
    do
    python main.py --env-name "CustomWalker2d-v0" \
    --xml walker2d_${xml}.xml
    done