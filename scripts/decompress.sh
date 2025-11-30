# 提前安装好pigz和pv
cd asset_data

# 解压asset的压缩包
pigz -dc imaginarium_assets.tar.gz \
  | pv -s $(stat -c%s imaginarium_assets.tar.gz) \
  | tar xf - -C ./

# 解压所有asset渲染图的压缩包
for file in imaginarium_assets_render_results_part*.tar.gz; do
  pigz -dc $file \
    | pv -s $(stat -c%s $file) \
    | tar xf - -C ./
done

# 解压asset patch embedding的压缩包
pigz -dc imaginarium_assets_patch_embedding.tar.gz \
  | pv -s $(stat -c%s imaginarium_assets_patch_embedding.tar.gz) \
  | tar xf - -C ./

# 解压所有asset 体素数据的压缩包
pigz -dc imaginarium_assets_voxels.tar.gz \
  | pv -s $(stat -c%s imaginarium_assets_voxels.tar.gz) \
  | tar xf - -C ./

# 解压背景贴图数据集的压缩包
pigz -dc background_texture_dataset.tar.gz \
  | pv -s $(stat -c%s background_texture_dataset.tar.gz) \
  | tar xf - -C ./


# 解压blender的压缩包
cd ../third_party/
pigz -dc blender-4.3.2-linux-x64.tar.gz \
   | pv -s $(stat -c%s blender-4.3.2-linux-x64.tar.gz) \
   | tar xf - -C ./




# 解压所有3d scene layout数据的压缩包
cd ../
for file in imaginarium_3d_scene_layout_dataset_part*.tar.gz; do
  pigz -dc $file \
    | pv -s $(stat -c%s $file) \
    | tar xf - -C ./
done