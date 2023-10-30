# author: xiao ran
# time: 2023-7-27
# using: 对长视频进行特征提取

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video_path = './Video/chew/#2_Gum_chew_h_nm_np1_fr_med_0.avi'
cap = cv2.VideoCapture(video_path)

frame_count = 0
action_counts = {}

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_count += 1

        # 对帧进行预处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = transforms.ToPILImage()(frame)
        input_frame = transform(pil_frame)
        input_frame = input_frame.unsqueeze(0)

        # 利用模型进行动作分类
        with torch.no_grad():
            output = model(input_frame)
            _, predicted_class = torch.max(output, 1)
            action = predicted_class.item()

            if action not in action_counts:
                action_counts[action] = 1
            else:
                action_counts[action] += 1
    else:
        break

cap.release()

total_frames = frame_count
action_frequencies = {}

for action, count in action_counts.items():
    frequency = count / total_frames
    action_frequencies[action] = frequency

print(action_frequencies)


