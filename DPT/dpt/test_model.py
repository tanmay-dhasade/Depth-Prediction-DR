import torch


from models import DPT,DPTDepthModel

import cv2


model_path = '/home/trdhasade/DR/DPT/weights/dpt_hybrid_kitti-cb926ef4.pt'

net_w = 1216
net_h = 352

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DPTDepthModel(
    path=model_path,
    scale=0.00006016,
    shift=0.00579,
    invert=True,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)

normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
)

model.eval()

if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)

img_path = '/home/trdhasade/DR/kitti_depth/depth/data_depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png'
img_input = cv2.imread(img_path)

with torch.no_grad():
    sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

    # if optimize == True and device == torch.device("cuda"):
    #     sample = sample.to(memory_format=torch.channels_last)
    #     sample = sample.half()

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    # if model_type == "dpt_hybrid_kitti":
    prediction *= 256

    # if model_type == "dpt_hybrid_nyu":
        # prediction *= 1000.0
    cv2.imwrite('test.png',prediction)