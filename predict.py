"""Predict native tree points from one RGB image; labels are never read."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dino-weights', type=Path, default=Path('weights/dinov2_vits14_pretrain.pth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new output directory.')
    os.environ['SQONET_PRETRAINED'] = '0'  # checkpoint supplies the backbone
    os.environ['SQONET_DINO_WEIGHTS'] = str(args.dino_weights.resolve())
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from paperlib import contracts as C, family as F
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    net = F.FamilyNet(oqs=True, qdpr=True, saf='S')
    net.load_state_dict(checkpoint['model'], strict=True)
    net.to(device).eval()
    bgr = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        p.error('Cannot decode image.')
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (800, 800), interpolation=cv2.INTER_LINEAR)
    a = (resized.astype(np.float32)/255 - np.asarray(C.MEAN, np.float32))/np.asarray(C.STD, np.float32)
    image = torch.from_numpy(a).permute(2, 0, 1).contiguous()[None].to(device)
    with torch.inference_mode():
        score, points, _ = net.predict(image, F.semantic_rgb(image))
        keep = score[0] > 0.5
        xy = points[0, keep].cpu().numpy()
        scores = score[0, keep].cpu().tolist()
    original = xy.astype(np.float64)*np.asarray([w/800, h/800])
    args.output.mkdir(parents=True)
    result = {'count': len(xy), 'coordinate_canvas': [800, 800], 'image_size': [w, h],
              'points_800': xy.tolist(), 'points_original': original.tolist(), 'scores': scores}
    (args.output/'points.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    vis = Image.fromarray(rgb)
    draw = ImageDraw.Draw(vis)
    radius = max(2, min(w, h)/250)
    for x, y in original:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='white')
    vis.save(args.output/'prediction.png')
    print(f'{len(xy)} predicted trees; saved to {args.output}')


if __name__ == '__main__':
    main()
