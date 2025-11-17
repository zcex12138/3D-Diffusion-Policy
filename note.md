# Pipeline
1. `gen_demo.py` 生成示范
2. `gen_pointcloud.py` 为示范生成点云（尚存在一点小问题，需要手动删掉data/image然后从原数据复制）
3. `gen_tactile_depth_img.py` 为示范生成触觉数据


# 注意
1. depth渲染出来后已经是物体尺度，而不是z-buffer
2. 如果点云在可视化时有大量原点附近的点，需要检查图片的size和sample的点云数量，如果size不够大则无法采样足够的点；不足的点会自动用零来填充
3. 