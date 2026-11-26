# Security policy

## Supported versions

Only the latest commit on `main` is supported.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/coopersimpson101-debug/ringtrain/security/advisories/new)
("Security" tab → "Report a vulnerability") with the affected commit, a description and a reproducer.
You should get an acknowledgement within 7 days; please allow up to 90 days for a fix before disclosure.

## Threat model

ringtrain is built for a **trusted home-lab LAN**. The peer and coordinator protocols are
**unauthenticated and unencrypted** plain TCP: anyone who can reach a worker's port can inject
gradients or abort a generation. Run it on a private network or a VPN (e.g. WireGuard) only.

What *is* defended:

- nothing received from the network is unpickled: frames are length-checked (2 GiB cap, magic,
  type) and control messages are JSON;
- checkpoints are loaded with `torch.load(..., weights_only=True)`, so a tampered checkpoint cannot
  execute code, and every blob's SHA-256 is checked against the `LATEST` pointer;
- the checkpoint store (raftstore) supports mutual TLS on its own side (see raftstore's docs);
  ringtrain's built-in HTTP client currently speaks plain HTTP only (roadmap).

In scope: code execution or file access through the frame parser, the rendezvous protocol, the
checkpoint loader or the CLI. Out of scope: attackers on the training LAN (see above) and
malicious adapters (an adapter is code you chose to run).
