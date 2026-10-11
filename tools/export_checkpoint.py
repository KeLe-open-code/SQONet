"""Strip optimizer/RNG metadata from a trusted local SQONet training checkpoint."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trusted-source', action='store_true', help='Confirm the source is your own trusted training checkpoint; legacy pickle loading can execute code.')
    args = p.parse_args()
    if not args.trusted_source:
        p.error('Only convert your own trusted checkpoints, with --trusted-source.')
    if args.output.exists():
        p.error('Output exists; choose a new filename.')
    import torch
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    state = source['model']
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        p.error('Expected a tensor model state dictionary.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': {k: v.detach().cpu() for k, v in state.items()}}, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
