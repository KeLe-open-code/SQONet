"""Train the DB-only SQONet chain using the released research stage functions."""
import argparse
import json
import os
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/DB'))
    p.add_argument('--image-root', type=Path)
    p.add_argument('--dino-weights', type=Path, default=Path('weights/dinov2_vits14_pretrain.pth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--arm', choices=[f'A{i}' for i in range(8)], default='A7')
    p.add_argument('--device', default='cuda')
    p.add_argument('--base-point-loss-coef', type=float, default=0.0001)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        p.error('Output already exists. Use a new directory or explicitly --resume.')
    os.environ['SQONET_DB_ROOT'] = str(args.data_root.resolve())
    os.environ['SQONET_IMAGE_ROOT'] = str((args.image_root or args.data_root/'images').resolve())
    os.environ['SQONET_DINO_WEIGHTS'] = str(args.dino_weights.resolve())
    os.environ['SQONET_PRETRAINED'] = '1'
    import numpy as np
    import torch
    from paperlib import contracts as C, data as D, stages as S, qdpr_train as Q
    from paperlib.trainloop import StageStore
    S.Device = Q.Device = torch.device(args.device)
    ds = D.load_dongbei(('train', 'val'))
    if D.verify_dataset(ds)['missing']:
        p.error('Missing authorized local images or annotations; see docs/DATA_FORMAT.md.')
    if args.arm in ('A3', 'A5', 'A6', 'A7') and not C.DINO_WEIGHT.is_file():
        p.error('Missing frozen DINOv2 ViT-S/14 weights.')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_file = args.output/'config.json'
    if args.resume and config_file.exists():
        old = json.loads(config_file.read_text())
        if any(old[k] != config[k] for k in ('seed', 'arm', 'base_point_loss_coef', 'data_root', 'image_root')):
            p.error('Resume settings differ from the saved run.')
    else:
        config_file.write_text(json.dumps(config, indent=2), encoding='utf8')

    def seed():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    def store(name):
        return StageStore(args.output/name)

    def best(name):
        root = args.output/name
        record = C.load_json(root/'best.json')
        path = root/record['checkpoint']
        assert C.sha256_file(path) == record['sha256']
        return path

    C.POINT_LOSS_COEF = args.base_point_loss_coef
    seed()  # before model construction, as in the multi-seed runner
    S.run_p2p_stage(store('A0'), ds, ds, 'train', 'val', args.seed, 'p2p_target', init_from=None)
    parent = best('A0')
    oqs = args.arm in ('A1', 'A4', 'A5', 'A7')
    qdpr = args.arm in ('A2', 'A4', 'A6', 'A7')
    saf = args.arm in ('A3', 'A5', 'A6', 'A7')
    C.POINT_LOSS_COEF = 0.0002
    if oqs:
        cache = store('oqs_cache')
        marker = cache.root/'cache/oqs_features.done'
        if not marker.exists():
            S.extract_oqs_cache(cache, parent, ds)
            marker.write_text('ok')
        seed()
        S.run_bqs_stage(store('BQS'), args.seed, cache_store=cache)
        seed()
        S.build_owned_sets(store('A1'), best('BQS'), cache_store=cache)
        S.run_owls_stage(store('A1'), args.seed, cache_store=cache, bqs_store=store('BQS'))
        parent = S.assemble_chain_ckpt(parent, best('BQS'), best('A1'), args.output/'chain_oqs.pt')
    if qdpr:
        name = 'A4' if oqs else 'A2'
        seed()
        Q.run_qdpr_stage_flip(store(name), ds, ds, args.seed, oqs_on=oqs, qdpr_on=True, parent_ckpt=parent)
        parent = best(name)
    if saf:
        seed()
        S.run_saf_stage(store(args.arm), ds, ds, args.seed, oqs_on=oqs, qdpr_on=qdpr,
                        saf_kind='S', parent_ckpt=parent, dino_store=store('dino_cache'))
        parent = best(args.arm)
    print(json.dumps({'arm': args.arm, 'checkpoint': str(parent), 'sha256': C.sha256_file(parent)}))


if __name__ == '__main__':
    main()
