from ultralytics import YOLO
# 导入cpca模块时,需要导入该模块,否则会异常报错
# import pandas._libs.tslibs.base
# import cpca
import pandas as pd
if __name__ == '__main__':
    # Create a new YOLO model from scratch

    # model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    model = YOLO(r'D:\yolo\ultralytics-main\ultralytics\cfg\models\v3\yolov3.yaml')
    # Load a pretrained YOLO model (recommended for training)
    # model = YOLO('runs/detect/train115/weights/best.pt')
    # model=YOLO('runs/detect/train115/weights/best.pt').load('yolov8n.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data=r'D:\yolo\ultralytics-main\ultralytics\cfg\datasets\VEDIA.yaml',epochs=200,
                          workers=4,batch=8,imgsz = 640,patience=500,cos_lr=True,optimizer='SGD'
                          )#,optimizer='Lion' optimizer='Sophia' ,optimizer='Sophia'#48
    # , imgsz = 1024
    # show_labels = False, show_conf = False

    # Evaluate the model's performance on the validation set

    # results = model.val(data='ultralytics/cfg/datasets/datawheathead_jianruo/GlobalWheat2020_mz_1.yaml')
    # results = model.predict(show=True, save=True)

    # results = model.val(data='ultralytics/cfg/datasets/GlobalWheat2020_mz.yaml', batch=1)#用于验证模型速度

    # Perform object detection on an image using the model

    # results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')
##


