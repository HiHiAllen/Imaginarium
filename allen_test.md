mkdir -p third_party
mkdir -p weights
mkdir -p asset_data

# 配置blender
huggingface-cli download binicey/Imaginarium-3D-Derived-Dataset \
    blender-4.3.2-linux-x64.tar.gz \
    --repo-type dataset \
    --local-dir ./third_party \
    --local-dir-use-symlinks False


pigz -dc ./third_party/blender-4.3.2-linux-x64.tar.gz | tar xvf - -C ./third_party/ | pv -l -s $(tar -tzf ./third_party/blender-4.3.2-linux-x64.tar.gz | wc -l) > /dev/null

sh blender_install.sh

sudo yum install -y libSM libXext libXrender libXi libXxf86vm mesa-libGL

# 配置权重
huggingface-cli download binicey/Imaginarium-3D-Derived-Dataset \
    ae_net_pretrained_weights.pth \
    depth_anything_v2_metric_hypersim_vitl.pth \
    dinov2_vitl14.pth \
    --repo-type dataset \
    --local-dir ./weights \
    --local-dir-use-symlinks False


# asset info
huggingface-cli download HiHiAllen/Imaginarium-Dataset \
    imaginarium_asset_info.csv \
    imaginarium_asset_info.xlsx \
    imaginarium_asset_info_with_render_images.xlsx \
    --repo-type dataset \
    --local-dir ./asset_data \
    --local-dir-use-symlinks False


# background_texture_dataset
huggingface-cli download HiHiAllen/Imaginarium-Dataset \
    background_texture_dataset.tar.gz \
    --repo-type dataset \
    --local-dir ./asset_data \
    --local-dir-use-symlinks False


pigz -dc ./asset_data/background_texture_dataset.tar.gz | tar xvf - -C ./asset_data/ | pv -l -s $(tar -tzf ./asset_data/background_texture_dataset.tar.gz | wc -l) > /dev/null



# asset info
huggingface-cli download HiHiAllen/Imaginarium-Dataset \
    imaginarium_3d_scene_layout_dataset_part1.tar.gz \
    imaginarium_3d_scene_layout_dataset_part2.tar.gz \
    imaginarium_3d_scene_layout_dataset_part3.tar.gz \
    imaginarium_3d_scene_layout_dataset_part4.tar.gz \
    --repo-type dataset \
    --local-dir ./asset_data \
    --local-dir-use-symlinks False

for f in ./asset_data/imaginarium_3d_scene_layout_dataset_part*.tar.gz; do
pigz -dc "$f" | tar xvf - -C ./asset_data/imaginarium | pv -l -s "$(tar -tzf "$f" | wc -l)" > /dev/null
done