import sys
sys.path.append("/media/cl/Data/2026/Baseline/vlmaps")
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.navigator.navigator import Navigator
from vlmaps.controller.discrete_nav_controller import DiscreteNavController
from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from vlmaps.map.map import Map
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf, base_pos2grid_id_3d, grid_id2base_pos_3d, base_rot_mat2theta
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np



import hydra
@hydra.main(version_base=None, config_path="/media/cl/Data/2026/Baseline/vlmaps/config/map_config", config_name="vlmaps.yaml")
def main(cfg):
    # 0. 初始化
    clmap=Map.create(cfg)
    clmap.load_map('/media/cl/Data/2026/Baseline/vlmaps/vlmaps_dataset/FloorPlan13_1')
    clmap.generate_obstacle_map() 
    clmap.init_categories(mp3dcat.copy()) # TODO 替换成ai2thor中的物体了
    controller = DiscreteNavController(cfg["controller_config"])
    vlmaps_dataloader = VLMapsDataloaderHabitat("/media/cl/Data/2026/Baseline/vlmaps/vlmaps_dataset/FloorPlan13_1", cfg, map=clmap)

    # 1. 计算在地图上面的初始点，输入应该包括pos和rot
    init_pos_thor=np.array([6.25,0.9009993672370911,4.5,0.0,0.7071067811865475,0.0,0.7071067811865476])
    base_tf = cvt_pose_vec2tf(init_pos_thor) # 转换为一个4*4的其次变换矩阵
    vlmaps_dataloader.from_habitat_tf(base_tf)
    init_pos_map = vlmaps_dataloader.to_full_map_pose()

    # 2. 计算目标位置点 【不需要朝向】
    pos=clmap.get_nearest_pos(init_pos_map[0:2],"Fridge")
    print(pos) #[441, 406]

    # 3. 计算路线（通过局部裁减计算，结合了避障-应该两点之间是直线）【不需要朝向】
    nav = Navigator()
    clmap.customize_obstacle_map(cfg.potential_obstacle_names,cfg.obstacle_names,vis=False)
    cropped_obst_map = clmap.get_customized_obstacle_cropped()
    nav.build_visgraph(cropped_obst_map,vlmaps_dataloader.rmin,vlmaps_dataloader.cmin,vis=False)
    paths = nav.plan_to(init_pos_map[0:2], pos, vis=False)  # take (row, col) in full map
    print(paths) # 这个是全局的路径点

    # 4. 计算动作序列 【需要朝向】每次转动为5度，每次前进为0.1
    actions_list, poses_list = controller.convert_paths_to_actions(init_pos_map, paths[1:]) # 两个坐标都是在地图上进行计算
    print(actions_list)
if __name__ == "__main__":
    main()