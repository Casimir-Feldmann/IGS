import os
import argparse
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow

from scipy import ndimage

def load_images(rgb_dirs, idx_1, idx_2):

    img1 = cv2.imread(rgb_dirs[idx_1])
    img2 = cv2.imread(rgb_dirs[idx_2])
    # img1 = cv2.cvtColor(img1 ,cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2 ,cv2.COLOR_BGR2RGB)

    return img1, img2

def calculate_flow(img1, img2, model, args):

    flow = inference_model(model, img1, img2)

    return flow
    
def save_mask(save_dir, mask):

    cv2.imwrite(save_dir, mask)

def color_segment(img, low=np.array([100, 0, 0]), high=np.array([255, 70, 70])):
    
    mask = cv2.inRange(img, low, high)

    return mask

def grabcut_segment(image, seed_mask):

    kernel = np.ones((3, 3), np.uint8) 
    erroded_mask = cv2.erode(seed_mask, kernel, iterations=2) 
    dilated_mask = cv2.dilate(seed_mask, kernel, iterations=2)

    mask = np.zeros(image.shape[:2],np.uint8)

    mask += np.where((dilated_mask - seed_mask) > 0, cv2.GC_PR_BGD, 0).astype('uint8')
    mask += np.where((seed_mask - erroded_mask) > 0, cv2.GC_PR_FGD, 0).astype('uint8')
    mask += np.where(erroded_mask > 0, cv2.GC_FGD, 0).astype('uint8')

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(image,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)

    mask_fg = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0)

    return mask_fg

def flow_segment(img_before, img_after, model, args):
    
    flow = calculate_flow(img_before, img_after, model, args)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

    mask = np.where(mag > args.flow_threshold, 1, 0)
    # mask = np.where(mag > args.flow_threshold, ang, 0)

    return mask

def flow_visualized(img_before, img_after, model, args):
    
    flow = calculate_flow(img_before, img_after, model, args)

    vis_flow = visualize_flow(flow)

    return vis_flow

def calc_gt_mask(image_before, image_after):

    mask_before_c = color_segment(image_before)
    mask_after_c = color_segment(image_after)

    mask_before_g = grabcut_segment(image_before, mask_before_c)
    mask_after_g = grabcut_segment(image_after, mask_after_c)

    diff = mask_after_g - mask_before_g

    mask = np.where(diff > 0, 1, 0)
    mask += np.where(diff < 0, -1, 0)

    return mask


def calc_static_statistics(image_list):

    pre_action = np.stack(image_list, axis=-1)
    mean = np.mean(pre_action, axis=-1)
    var = np.var(pre_action, axis=-1)

    return mean, var

def calc_image_distance(mean, var, current_image, eps=0.0001):

    var_norm = var / var.max()
    dist_adjusted = (current_image - mean) / (var_norm + eps)
    dist = np.linalg.norm(dist_adjusted, axis=-1)

    return dist

def filter_noise(mean_mask, connected_size_threshold=100):

    Zlabeled, Nlabels = ndimage.label(mean_mask)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label,size in enumerate(label_size):
        if size < connected_size_threshold:
            mean_mask[Zlabeled == label] = 0

    return mean_mask


def main(args):

    rgb_dirs = [os.path.join(args.rgb_path, file) for file in sorted(os.listdir(args.rgb_path))]

    device = 'cuda:0'
    model = init_model(config=args.flow_config_path, 
                       checkpoint=args.flow_checkpoint_path, 
                       device=device)
    
    # pre_motion_idx = 250
    # dist_threshold = 250
    # connected_size_threshold = 100

    # For 2023-10-27-13-20-44
    # 0,618,1013,1454,1600
    # For bagfiles_casi/automated_sweep_2
    # 250, 1000

    # pre_action_list = []
    # for idx in range(min((len(rgb_dirs)-1), pre_motion_idx)):
    #     pre_action_list.append(cv2.imread(rgb_dirs[idx]))
    # scene_mean, scene_var = calc_static_statistics(pre_action_list)

    # idx = 1000
    # post_action_image = cv2.imread(rgb_dirs[idx])

    # dist = calc_image_distance(scene_mean, scene_var, post_action_image)

    # mean_mask = np.where(dist > dist_threshold, 1, 0)
    # mean_mask = filter_noise(mean_mask, connected_size_threshold)

    # plt.imshow(mean_mask)
    # plt.imshow(post_action_image/255)
    # plt.imshow(mean_mask, alpha=0.2)
    # plt.show()
    # plt.savefig(f"{args.save_path}/mask_{idx}.png")
    # plt.close()
    # save_mask(f"{args.save_path}/mask_{idx}.png", mean_mask*255)


    for idx in tqdm(range(len(rgb_dirs)-1)):
        
        # image_before, image_after = load_images(rgb_dirs, idx, idx+1)
        image_before, image_after = load_images(rgb_dirs, 250, 1000)

        flow = calculate_flow(image_before, image_after, model, args)
        flow_vis = visualize_flow(flow)
        mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        mask = np.where(mag > args.flow_threshold, 1, 0)

        # plt.imshow(mask)
        # plt.imshow(image_before, alpha=0.4)
        # plt.show()
        # plt.savefig(f"{args.save_path}/flow_mask.png")
        # plt.close()

        # break

        # mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

        # mag_mask = np.where(mag > args.flow_threshold, 1, 0).reshape((flow_vis.shape[0], flow_vis.shape[1]))
        # mag_mask = np.where(mag > args.flow_threshold, 1, 0)
        
        # plt.imshow(flow_vis)
        # plt.show()

        # breakpoint()
        # h = flow.shape[0]
        # w = flow.shape[1]
        # flow[:,:,0] += np.arange(w)
        # flow[:,:,1] += np.arange(h)[:,np.newaxis]
        
        # new_frame = cv2.remap(image_before, flow, None, cv2.INTER_LINEAR)
        # plt.imshow(new_frame-image_before)
        # plt.show()

        # plt.imshow(new_frame)
        # plt.show()


        # plt.imshow(image_before)
        # plt.imshow(mag / mag.max(), alpha=0.4)
        # plt.show()

        # plt.imshow(image_after)
        # plt.imshow(mean_mask, alpha=0.4)
        # plt.show()
        
        # flow_vis_mag = flow_vis * mag_mask[..., np.newaxis]
        # print(flow_vis_mag.shape)
        # plt.imshow(flow_vis_mag)
        # plt.show()


        # image_before, image_after = load_images(rgb_dirs, idx, idx+5)


        # mask = flow_segment(image_before, image_after, model, args)

        # gt = calc_gt_mask(image_before, image_after)

        # plt.imshow(image_before)
        # plt.imshow(gt, alpha=0.4)
        # plt.imshow(mask, alpha=0.6)
        # plt.show()
        # plt.savefig(f"{args.save_path}/mask_{idx}.png")
        # plt.close()

        # plt.imshow(image_after)
        # plt.imshow(mask, alpha=0.5)
        # plt.imshow(gt, alpha=0.5)
        # plt.show()


        # flow_vis = flow_visualized(image_before, image_after, model, args)
        # save_mask(f"{args.save_path}/mask_{idx}.png", flow_vis)

        mask *= 255
        save_mask(f"{args.save_path}/{idx:05d}_mask.png", mask)


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create binary segmentation masks")
    parser.add_argument("--flow-checkpoint-path", dest="flow_checkpoint_path", required=True, help="Path to the flow checkpoint")
    parser.add_argument("--flow-config-path", dest="flow_config_path", required=True, help="Path to the flow configuration")
    parser.add_argument("--rgb-path", dest="rgb_path", required=True, help="Path to rgb directory")
    parser.add_argument("--save-path", dest="save_path", required=True, help="Path to save masks")
    parser.add_argument("--eval-path", dest="eval_path", required=True, help="Path to gt masks")
    parser.add_argument("--flow-threshold", dest="flow_threshold", required=True, help="Minimum flow magnitude threshold", type=float)

    args = parser.parse_args()

    main(args)