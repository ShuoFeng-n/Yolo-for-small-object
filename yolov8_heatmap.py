# import warnings
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# import torch, yaml, cv2, os, shutil
# import numpy as np
#
# np.random.seed(0)
# import matplotlib.pyplot as plt
# from tqdm import trange
# from PIL import Image
# from ultralytics.nn.tasks import DetectionModel as Model
# from ultralytics.utils.torch_utils import intersect_dicts
# # from ultralytics.yolo.data.augment import LetterBox
# from ultralytics.utils.ops import xywh2xyxy
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
#
#
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)
#
#
# class yolov8_heatmap:
#     def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
#         device = torch.device(device)
#         ckpt = torch.load(weight)
#         model_names = ckpt['model'].names
#         csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
#         model = Model(cfg, ch=3, nc=len(model_names)).to(device)
#         csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
#         model.load_state_dict(csd, strict=False)  # load
#         model.eval()
#         print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
#
#         target_layers = [eval(layer)]
#         method = eval(method)
#
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
#         self.__dict__.update(locals())
#
#     def post_process(self, result):
#         logits_ = result[:, 4:]
#         boxes_ = result[:, :4]
#         sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#         return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
#             indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
#
#     def draw_detections(self, box, color, name, img):
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
#                     lineType=cv2.LINE_AA)
#         return img
#
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)
#
#         # img process
#         img = cv2.imread(img_path)
#         img = letterbox(img)[0]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#
#         # init ActivationsAndGradients
#         grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
#
#         # get ActivationsAndResult
#         result = grads(tensor)
#         activations = grads.activations[0].cpu().detach().numpy()
#
#         # postprocess to yolo output
#         post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf_threshold:
#                 break
#
#             self.model.zero_grad()
#             # get max probability for this prediction
#             if self.backward_type == 'class' or self.backward_type == 'all':
#                 score = post_result[i].max()
#                 score.backward(retain_graph=True)
#
#             if self.backward_type == 'box' or self.backward_type == 'all':
#                 for j in range(4):
#                     score = pre_post_boxes[i, j]
#                     score.backward(retain_graph=True)
#
#             # process heatmap
#             if self.backward_type == 'class':
#                 gradients = grads.gradients[0]
#             elif self.backward_type == 'box':
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
#             else:
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
#                             grads.gradients[4]
#             b, k, u, v = gradients.size()
#             weights = self.method.get_cam_weights(self.method, None, None, None, activations,
#                                                   gradients.detach().numpy())
#             weights = weights.reshape((b, k, 1, 1))
#             saliency_map = np.sum(weights * activations, axis=1)
#             saliency_map = np.squeeze(np.maximum(saliency_map, 0))
#             saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
#             saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#             if (saliency_map_max - saliency_map_min) == 0:
#                 continue
#             saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#
#             # add heatmap and box to image
#             cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
#             # cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
#             #                                  f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
#             #                                  cam_image)
#             # cam_image = self.draw_detections(post_boxes[100], self.colors[int(post_result[i, :].argmax())],
#             #                                  f'1 ',
#             #                                  cam_image)
#             # 上面这句话的意思是给图片加标注框
#             cam_image = Image.fromarray(cam_image)
#             cam_image.save(f'{save_path}/{i}.png')
#
#         # 'weight': 'runs/detect/train175/weights/best.pt',
#         # 'cfg': 'ultralytics/cfg/models/v8/xiaomai_model/head_test.yaml',
#
#         # runs/detect/train144/weights/last.pt
#         # ultralytics/cfg/models/v8/yolov8.yaml
#
#         # runs / detect / train176 / weights / last.pt
#         # ultralytics / cfg / models / v5 / yolov5.yaml
#
#         # ultralytics/cfg/models/v8/xiaomai_model/ContextAggregation.yaml
#         # 'runs/detect/train148/weights/best.pt'
# def get_params():
#     params = {
#         'weight': 'runs/detect/train242/weights/last.pt',
#         'cfg': 'ultralytics/cfg/models/v8/yolov8.yaml',
#         'device': 'cuda:0',
#         'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
#         'layer': 'model.model[9]', #'model.model[9]'
#         'backward_type': 'all',  # class, box, all
#         'conf_threshold': 0.6,  # 0.6
#         'ratio': 0.02  # 0.02-0.1
#     }
#     return params
# # #runs/detect/train214/weights/last.pt
# #ultralytics/cfg/models/v8/goolenet-cspdarknet/MzNet.yaml
#
# #  runs/detect/train144/weights/last.pt
# # ultralytics/cfg/models/v8/yolov8.yaml
#
# #         'weight': 'runs/detect/train148/weights/best.pt',
# #         'cfg': 'ultralytics/cfg/models/v8/xiaomai_model/ContextAggregation.yaml',
#
# #         'weight': 'runs/detect/train176/weights/best.pt',
# #         'cfg': 'ultralytics/cfg/models/v5/yolov5.yaml',
#
# #         'weight': 'runs/detect/train160/weights/last.pt',
# #         'cfg': 'ultralytics/cfg/models/v8/xiaomai_model/C2f_DySnakeConv.yaml',
#
# #         'weight': 'runs/detect/train164/weights/last.pt',
# #         'cfg': 'ultralytics/cfg/models/v8/yolov8_space_to_depth.yaml',
#
# #         'weight': 'runs/detect/train190/weights/last.pt',
# #         'cfg': 'ultralytics/cfg/models/jiangwei/yolov8.yaml',
# if __name__ == '__main__':
#     model = yolov8_heatmap(**get_params())
#     model(r'dataesseay-wheat/piping/DJI_20240704143448_0001_V.JPG', 'sc')
#    # model(r'dataesseay-wheat/piping/DJI_20240704143449_0002_V.JPG', 'sc')
#    # model(r'dataesseay-wheat/piping/DJI_20240704143456_0006_V.JPG', 'sc')
#     # model(r'datawheathead_heat/4caebe5ac.jpg', 'sc')dataesseay-wheat/test/20-enhance (1).png
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
import torch.serialization
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.yolo.data.augment import LetterBox
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):

        device = torch.device(device)

        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        ckpt = torch.load(weight, weights_only=False)
        ckpt = torch.load(weight, weights_only=False)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    # def draw_detections(self, box, color, name, img):
    #     xmin, ymin, xmax, ymax = list(map(int, list(box)))
    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
    #     cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
    #                 lineType=cv2.LINE_AA)
    #     return img
    # 在 draw_detections 中加粗检测框
    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))

        # 使用黑色边框 (0,0,0)，线宽加粗到3
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), thickness=3)  # 黑色边框

        # 白底黑字（增强对比度）
        text = f"{name}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (xmin, ymin - text_height - 10), (xmin + text_width, ymin - 5), (255, 255, 255), -1)  # 白底
        cv2.putText(img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # 黑字

        return img

    # def __call__(self, img_path, save_path):
    #     # remove dir if exist
    #     if os.path.exists(save_path):
    #         shutil.rmtree(save_path)
    #     # make dir if not exist
    #     os.makedirs(save_path, exist_ok=True)
    #
    #     # img process
    #     img = cv2.imread(img_path)
    #     img = letterbox(img)[0]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.float32(img) / 255.0
    #     tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
    #
    #     # init ActivationsAndGradients
    #     grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
    #
    #     # get ActivationsAndResult
    #     result = grads(tensor)
    #     activations = grads.activations[0].cpu().detach().numpy()
    #
    #     # postprocess to yolo output
    #     post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
    #     for i in trange(int(post_result.size(0) * self.ratio)):
    #         if float(post_result[i].max()) < self.conf_threshold:
    #             break
    #
    #         self.model.zero_grad()
    #         # get max probability for this prediction
    #         if self.backward_type == 'class' or self.backward_type == 'all':
    #             score = post_result[i].max()
    #             score.backward(retain_graph=True)
    #
    #         if self.backward_type == 'box' or self.backward_type == 'all':
    #             for j in range(4):
    #                 score = pre_post_boxes[i, j]
    #                 score.backward(retain_graph=True)
    #
    #         # process heatmap
    #         if self.backward_type == 'class':
    #             gradients = grads.gradients[0]
    #         elif self.backward_type == 'box':
    #             gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
    #         else:
    #             gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
    #                         grads.gradients[4]
    #         b, k, u, v = gradients.size()
    #         weights = self.method.get_cam_weights(self.method, None, None, None, activations,
    #                                               gradients.detach().numpy())
    #         weights = weights.reshape((b, k, 1, 1))
    #         saliency_map = np.sum(weights * activations, axis=1)
    #         saliency_map = np.squeeze(np.maximum(saliency_map, 0))
    #         saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
    #         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    #         if (saliency_map_max - saliency_map_min) == 0:
    #             continue
    #         saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
    #
    #         # add heatmap and box to image
    #         cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
    #         cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
    #                                          f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
    #                                          cam_image)
    #         cam_image = self.draw_detections(post_boxes[100], self.colors[int(post_result[i, :].argmax())],
    #                                          f'1 ',
    #                                          cam_image)
    #         # 上面这句话的意思是给图片加标注框
    #         cam_image = Image.fromarray(cam_image)
    #         cam_image.save(f'{save_path}/{i}.png')
    def __call__(self, img_path, save_path):
        # 移除旧目录并创建新目录
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # 读取并预处理图片
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 用于绘制检测框
        img_float = np.float32(img_rgb) / 255.0  # 用于热力图叠加
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 初始化 ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # 获取模型输出
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # 后处理，获取预测框和类别
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])

        # 初始化总热力图
        total_saliency_map = np.zeros((tensor.size(2), tensor.size(3)), dtype=np.float32)

        # 遍历所有检测到的目标
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break

            self.model.zero_grad()

            # 计算梯度（class/box/all）
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # 计算当前目标的热力图
            gradients = grads.gradients[0]  # 默认仅使用 class 梯度
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))

            # 累加到总热力图
            total_saliency_map += saliency_map

            # 绘制检测框和类别标签（直接绘制到 img_rgb 上）
            box = post_boxes[i]
            class_id = int(post_result[i, :].argmax())
            class_name = self.model_names[class_id]
            confidence = float(post_result[i].max())
            color = self.colors[class_id]
            img_rgb = self.draw_detections(box, color, f"{class_name} {confidence:.2f}", img_rgb)

        # 归一化总热力图
        total_saliency_map = (total_saliency_map - total_saliency_map.min()) / (
                    total_saliency_map.max() - total_saliency_map.min() + 1e-8)

        # 将热力图叠加到原图（此时 img_rgb 已包含所有检测框）
        cam_image = show_cam_on_image(img_float, total_saliency_map, use_rgb=True)

        # 将检测框从 img_rgb 叠加到热力图上（避免热力图覆盖检测框）
        # 因为 show_cam_on_image 会改变图像颜色，我们需要重新叠加检测框
        # cam_image = cv2.addWeighted(
        #     cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), 0.8,
        #     cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.2, 0
        # )
        #
        # # 保存最终结果
        # cv2.imwrite(f"{save_path}/final_heatmap.png", cam_image)
        # 在生成热力图后，用加权叠加保留检测框
        cam_image = show_cam_on_image(img_float, total_saliency_map, use_rgb=True, image_weight=0.5)  # 热力图较强
        # 将原始检测框（img_rgb）以50%透明度叠加回去
        final_image = cv2.addWeighted(
            cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR), 0.7,
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.3, 0
        )
        cv2.imwrite(f"{save_path}/final_heatmap.png", final_image)


def get_params():
    params = {
        'weight': r'D:\yolo\ultralytics-main\ultralytics\runs\detect\train29\weights\best.pt',
        'cfg': r'D:\yolo\ultralytics-main\ultralytics\cfg\models\v8\yolov8_gam.yaml',
         'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 'model.model[9]',  # 'model.model[9]'
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.3,  # 0.6
        'ratio': 0.02  # 0.02-0.1
    }
    return params
# "D:\yolo\ultralytics-main\ultralytics\sc2"

if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'D:\yolo\ultralytics-main\image\微信图片_20250330135135.jpg', 'sc2')