# Deploy

## systemd (user service)

`card-scanner-siglip.service` runs the API under the user's own systemd
instance (`systemctl --user`), so it doesn't need root and stays tied to
the user's own environment/venv.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/card-scanner-siglip.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now card-scanner-siglip.service
```

**For it to actually survive a reboot** (not just a logout/login), the
user needs lingering enabled -- otherwise a user-level systemd instance
only starts once that user logs in, not at boot:

```bash
loginctl enable-linger "$(whoami)"
```

Check it's running:

```bash
systemctl --user status card-scanner-siglip.service
curl http://localhost:8000/health
```

Adjust `WorkingDirectory`/`ExecStart` in the unit file if the repo or
venv live somewhere other than `/home/user/projects/card-scanner-siglip`.
`Restart=on-failure` means a crash gets retried automatically (10s
backoff); it won't mask a genuinely broken deploy, since a crash-loop
still shows up in `systemctl --user status` / `journalctl --user -u
card-scanner-siglip`.
