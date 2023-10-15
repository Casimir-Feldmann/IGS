from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow

import matplotlib.pyplot as plt

config_file = '/home/casimir/ETH/SemesterProject/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
checkpoint_file = '/home/casimir/ETH/SemesterProject/mmflow/checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
# img1 = '/home/casimir/ETH/SemesterProject/mmflow/demo/frame_0001.png'
# img2 = '/home/casimir/ETH/SemesterProject/mmflow/demo/frame_0002.png'

img1 = '/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/rgb/00001_rgb.png'
img2 = '/home/casimir/ETH/SemesterProject/IGS/dataset_easy/stop_motion_1_processed/rgb/00002_rgb.png'

result = inference_model(model, img1, img2)

write_flow(result, flow_file='flow.flo')

flow_map = visualize_flow(result, save_file='flow_map.png')

plt.imshow(flow_map)
plt.show()