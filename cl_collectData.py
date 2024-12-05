from ai2thor.controller import Controller
import prior
import numpy as np 
from scipy.spatial.transform import Rotation as R
from PIL import Image
import random
from tqdm import tqdm
from prior import LazyJsonDataset, DatasetDict
import json
def euler_to_quaternion(euler_angles):
    """
    将欧拉角 (x, y, z) 转换为四元数 (qx, qy, qz, qw)
    参数:
    euler_angles: dict 包含 'x', 'y', 'z' 旋转角度，以度为单位
    返回:
    四元数的数组 (qx, qy, qz, qw)
    """
    # 以度数为单位时可以先转换为弧度 np.radians()
    roll = np.deg2rad(euler_angles['x'])
    pitch = np.deg2rad(euler_angles['y'])
    yaw = np.deg2rad(euler_angles['z'])
    # 使用 scipy.spatial.transform.Rotation
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    quaternion = r.as_quat()  # [qx, qy, qz, qw] 的格式
    return list(quaternion)

def get_object_indices(seg_map, semantic_to_id, objectlist):
    # 初始化结果数组，使用 -1 表示不在 objectlist 中的像素
    h, w, _ = seg_map.shape
    indices_map = 0 * np.ones((h, w), dtype=int)
    
    # 遍历每个像素点
    for i in range(h):
        for j in range(w):
            # 获取当前像素点的 RGB 值
            pixel = tuple(seg_map[i, j])
            
            # 检查该 RGB 是否在字典中
            if pixel in list(semantic_to_id.keys()):
                # 获取物体 ID 名称，去掉 ID 部分
                full_name = semantic_to_id[pixel]
                object_name = full_name.split('|')[0]  # 获取物体名称
                
                # 检查物体名称是否在 objectlist 中，找到其下标
                if object_name in objectlist:
                    object_index = objectlist.index(object_name)
                    indices_map[i, j] = object_index  # 设置类别的下标
    return indices_map
def load_dataset() -> prior.DatasetDict:
    data = {}
    for split, size in [("test", 1000)]:
        # 使用 open 以文本模式读取 .jsonl 文件
        with open('/media/cl/Data/2026/Baseline/vlmaps/test.jsonl', "r", encoding="utf-8") as f:
            houses = [line.strip() for line in tqdm(f, total=size, desc=f"Loading {split}")]  # 去除每行的换行符
        # 创建 LazyJsonDataset
        data[split] = LazyJsonDataset(data=houses, dataset="procthor-dataset", split=split)
    return DatasetDict(**data)

dataset=load_dataset()
house = dataset["test"][13]
c = Controller(scene=house, 
               quality='High WebGL',
               agentMode="locobot",
               gridSize=0.25, 
               snapToGrid=False,rotateStepDegrees=15,
               renderDepthImage=True, renderClassImage=True, renderObjectImage=True, renderImage=True, renderInstanceSegmentation=True,
               width=1080, height=720, 
               fieldOfView=90,
               )
c.step(
    action="Teleport",
    horizon=0
)
with open('/media/cl/Data/2026/Baseline/vlmaps/action.json','r') as f: actions=json.load(f)

myLabel=['Void','Floor','Door','Wall','AluminumFoil', 'ArmChair', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Cabinet', 'Chair', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'Curtains', 'Desk', 'Desktop', 'DiningTable', 'DogBed', 'Drawer', 'Dresser', 'Faucet', 'Floor', 'FloorLamp', 'Footstool', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowelHolder', 'LaundryHamper', 'LightSwitch', 'Microwave', 'Mirror', 'Poster', 'PotatoSliced*', 'Safe', 'ScrubBrush', 'Shelf', 'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'Sofa', 'Stool', 'StoveBurner', 'StoveKnob', 'TargetCircle', 'Television', 'Toaster', 'Toilet', 'ToiletPaperHanger','Towel', 'TowelHolder', 'TVStand', 'VacuumCleaner', 'Window',]
NUM=0
root="/media/cl/Data/2026/Baseline/vlmaps/vlmaps_dataset/FloorPlan13_3"
for a in actions:

    NUM+=1
    event = c.step(action=a)
    # if NUM>107:
    if event.metadata["lastActionSuccess"]:
        print(f"动作{a}成功")
        depth = c.last_event.depth_frame
        np.save(f"{root}/depth/FloorPlan13_{NUM}.npy", depth)

        rgb=Image.fromarray(c.last_event.frame)
        rgb.save(f"{root}/rgb/FloorPlan13_{NUM}.png")
        
        position=[i[1] for i in c.last_event.metadata['agent']['position'].items()]
        quaternion= euler_to_quaternion(c.last_event.metadata['agent']['rotation'])
        pose=' '.join(map(str, position+quaternion))
        with open(f"{root}/pose/FloorPlan13_{NUM}.txt", 'w') as file:file.write(pose)

        # semantic_raw=c.last_event.instance_segmentation_frame
        # label=c.last_event.color_to_object_id
        # semantic=get_object_indices(semantic_raw,label,myLabel)
        # np.save(f"{root}/semantic/FloorPlan13_{NUM}.npy", semantic)
    else:
        print("失败！！！！！！！！！！！！！")

    


    