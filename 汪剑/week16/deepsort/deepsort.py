from ultralytics import YOLO
import cv2
import torch

# 加载YOLO模型
yolo_model = YOLO('yolov5s.pt')


class DeepSort:
    # 初始化 Deepsort 跟踪器
    def __init__(self):
        self.trackers = []  # 存储当前跟踪器检测到的目标边框信息

    # 定义更新跟踪器的方法
    def update(self, detections):
        '''
        :param detections: 本帧检测到的目标列表（边框信息）
        :return:
        '''
        confirmed_tracks = []  # 存储确认匹配的目标边框
        for det in detections:
            matched = False  # 当前检测目标尚未匹配到已存在的跟踪器
            for i, trk in enumerate(self.trackers):
                # 简单距离匹配，这里简化为中心坐标距离
                # 计算当前检测框的中心点坐标
                center_det = [det[0] + det[2] / 2, det[1] + det[3] / 2]
                # 计算当前跟踪器框的中心点坐标
                center_trk = [trk[0] + trk[2] / 2, trk[1] + trk[3] / 2]
                # 欧氏距离公式计算检测框中心和跟踪器框中心之间的距离
                dist = ((center_det[0] - center_trk[0]) ** 2 + (center_det[1] - center_trk[1]) ** 2) ** 0.5
                # 距离小于某个阈值（此处设定为 50）则认为两个框对应的是同一个目标
                if dist < 50:
                    self.trackers[i] = det
                    confirmed_tracks.append(det)
                    matched = True
                    break
            if not matched:
                self.trackers.append(det)
        return confirmed_tracks


# 打开视频文件
cap = cv2.VideoCapture('test5.mp4')
tracker = DeepSort()

while cap.isOpened():
    ret, frame = cap.read()  # 读取下一帧视频，ret 为布尔值表示是否读取成功，frame 为当前帧的图像数据
    if not ret:
        break

    # 使用 yolo 进行目标检测
    results = yolo_model(frame)
    detections = []  # 存储当前帧的目标检测框
    for box in results[0].boxes:  # 遍历检测结果中第一张图（通常 batch_size=1）的所有目标框
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # box.xyxy 数据格式：tensor([[x1, y1, x2, y2]])
        conf = box.conf.item()  # 获取该检测框的
        # 置信度，并转换为 Python 数值
        if conf > 0.5:
            detections.append([x1, y1, x2 - x1, y2 - y1])

    # 使用 DeepSort 进行跟踪
    tracker_objects = tracker.update(detections)

    # 绘制跟踪结果
    for obj in tracker_objects:
        x1,y1,w,h = obj
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(0,255,0),2)

    cv2.imshow('Traffic Tracking',frame)
    '''
    cv2.waitKey() 可能在某些系统上返回 32 位整数，而 ord() 返回 8 位整数。
    通过按位与运算 & 0xFF 只保留 waitKey() 返回值的低 8 位，避免不必要的干扰
    
    key = cv2.waitKey(1)  # 可能返回 32 位整数，如 0x00000071
    key = key & 0xFF      # 变为 0x71，即 113
    '''
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release() # 释放视频捕获对象，关闭视频文件
cv2.destroyAllWindows() # 关闭所有由 OpenCV 打开的窗口，清理资源

