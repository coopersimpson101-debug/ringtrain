## What does this change?

<!-- One or two sentences. Link the issue if there is one. -->

## How is it tested?

- [ ] `python -m pyflakes ringtrain tests scripts`
- [ ] `python -m pytest -q -m "not slow and not cluster"`
- [ ] `python -m pytest -q -m "slow or cluster"` (if transport, rendezvous, elastic or checkpoint code changed)
- [ ] a new or updated test that fails without this change

## Checklist

- [ ] no `torch.distributed` / MPI / other communication library
- [ ] wire format and checkpoint layout unchanged, or the change is described
- [ ] README numbers updated only from a reproducible run in `results/`
