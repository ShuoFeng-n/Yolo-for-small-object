# from ultralytics import YOLO
# from PIL import Image
# import cv2
#
# # model = YOLO("runs/detect/train175/weights/best.pt")#ours
# # model = YOLO("runs/detect/train144/weights/last.pt")#yolov8
# # model = YOLO("runs/detect/train166/weights/last.pt")#93.6 48.0
# model = YOLO(r"D:\yolo\ultralytics-main\ultralytics\runs\detect\train29\weights\best.pt")#94.1 49.5
# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# # results = model.predict(source="0")
# # results =model.track(source="global-wheat-detection/test",show=True,save=True)
# results = model.predict(source=r"D:\yolo\ultralytics-main\e5502927d6eaa87bef24cd738e133de.jpg", save=True) # Display preds. Accepts all YOLO predict arguments
from ultralytics import YOLO
import os
import shutil

# 初始化模型
model = YOLO(r"D:\yolo\ultralytics-main\ultralytics\runs\detect\train29\weights\best.pt")

# 输入输出配置
input_folder = r"D:\yolo\ultralytics-main\image"  # 需要预测的图片文件夹路径
output_folder = r"D:\yolo\ultralytics-main\detect_results1"  # 结果保存路径

# 清空并创建输出文件夹
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# 执行批量预测（自动保存带检测框的图片）
results = model.predict(
    source=input_folder,
    save=True,                # 自动保存检测结果
    save_txt=False,           # 不保存标签文件
    save_conf=False,          # 不保存置信度
    save_crop=False,          # 不保存裁剪目标
    project=output_folder,    # 指定保存路径
    name='',                  # 不创建子文件夹
    exist_ok=True,            # 允许覆盖已有文件
                    # 推理尺寸
    conf=0.1,                 # 置信度阈值
    device='0',               # 使用GPU 0
               # 最大检测目标数
    visualize=False,          # 禁用特征可视化
    augment=False,            # 禁用测试时增强
)

# 将结果文件从默认位置移动到指定目录（适用于旧版本YOLO）
# 新版本直接使用 project 参数即可
print(f"预测完成，结果已保存至：{output_folder}")