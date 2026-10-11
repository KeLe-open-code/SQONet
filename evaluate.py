"""Evaluate a released SQONet checkpoint on an explicitly selected DB split."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/DB'))
    p.add_argument('--image-root', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dino-weights', type=Path, default=Path('weights/dinov2_vits14_pretrain.pth'))
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; choose a new directory.')
    os.environ.update(SQONET_PRETRAINED='0', SQONET_DB_ROOT=str(args.data_root.resolve()),
                      SQONET_IMAGE_ROOT=str((args.image_root or args.data_root/'images').resolve()),
                      SQONET_DINO_WEIGHTS=str(args.dino_weights.resolve()))
    import torch
    from paperlib import data as D, family as F, stages as S, eval as E
    S.Device = torch.device(args.device)
    ds = D.load_dongbei((args.split,))
    net = F.FamilyNet(oqs=True, qdpr=True, saf='S')
    net.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    net.to(S.Device).eval()
    metrics, rows, _ = S.val_family(net, ds, args.split)
    args.output.mkdir(parents=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in E.rows_to_jsonl(rows)), encoding='utf8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
