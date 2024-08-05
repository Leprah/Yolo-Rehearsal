from ultralytics import YOLO

model = YOLO("yolov8n-cls.yaml")
results = model.train(data="/home/kevin/leprah/dataset", epochs=2, imgsz=640, rehearsal_replay_paths=['/home/kevin/leprah/dataset1'], device='cuda')

# import torchvision
# x = torchvision.datasets.ImageFolder('/home/kevin/leprah/dataset/train')
# y = torchvision.datasets.ImageFolder('/home/kevin/leprah/dataset/val')
# xy = x+y
# print(x.root)

# for i in x:
#     print(i)
