# wit-stage1-nix-ci

> **Local CI pipeline for Weak Inhomogeneity Theory – Stage 1**
> powered by **GitLab Runner (shell executor)** & **Nix flake check** on MacBook Pro M3 Max.

---

![build](https://gitlab.com/your-group/wit-stage1-nix-ci/badges/main/pipeline.svg)

This repository hosts the automation scripts, Nix definitions, and GitLab CI configuration required to run reproducible, container‑less continuous integration for the Stage 1 experiments of the *Weak Inhomogeneity Theory* (WIT) project.

* **GPU‑ready, no Docker:** tests execute directly on Apple Silicon/Metal or any host GPU.
* **Single‑command validation:** `nix flake check` rebuilds and runs all tests with locked dependencies.
* **Identical CI & local flow:** the `.gitlab-ci.yml` simply calls the same `nix flake check`, so whatever passes locally will pass on GitLab.
* **Self‑contained:** every dependency is pinned via `flake.lock`; rolling back or upgrading is deterministic.

## 📦 Repository layout

```text
.gitlab-ci.yml        # GitLab runner pipeline (shell executor)
flake.nix             # Nix flake root – inputs & outputs
flake.lock            # Fully pinned dependency graph
nix/
  ci.nix              # Dev‑shell & test definitions
scripts/
  run_stage1.py       # Stage 1 orchestration script
LICENSE               # MIT license text
README.md             # You are here
```

## 🚀 Quick start

```bash
# 1 — Install prerequisites (macOS/Linux)
curl -L https://nixos.org/nix/install | sh -- --daemon
brew install gitlab-runner direnv pre-commit

# 2 — Clone & enter dev‑shell
git clone https://gitlab.com/your-group/wit-stage1-nix-ci.git
cd wit-stage1-nix-ci
direnv allow           # loads the nix shell automatically

# 3 — Run the full test suite locally
nix flake check

# 4 — Mimic CI with GitLab Runner
gitlab-runner exec shell flake-check
```

> **Tip:** If you push to a GitLab project that has a runner tagged `m3max` (or any custom tag you configure) the exact same pipeline will run server‑side.

## 🔧 Development workflow

1. Edit source (`scripts/`, `nix/`, etc.).
2. `nix flake update` *(optional)* to upgrade dependencies.
3. `nix flake check` to ensure everything still passes.
4. Commit & push.  CI will replicate the local result.

### Common tasks

| Purpose                      | Command                                        |
| ---------------------------- | ---------------------------------------------- |
| Enter dev shell              | `nix develop`                                  |
| Clean old store generations  | `nix-collect-garbage -d`                       |
| Run single test file         | `python scripts/run_stage1.py --case <name>`   |
| Register local runner (once) | `sudo gitlab-runner register --executor shell` |

## 🐛 Troubleshooting

| Symptom                        | Fix                                                   |
| ------------------------------ | ----------------------------------------------------- |
| `nix flake check` very slow    | `nix-store --gc` then enable `/nix/store` cache in CI |
| `command not found: nix` in CI | Add Nix profile path to runner `config.toml`          |
| Metal backend crash (JAX)      | Disable X64: `export JAX_ENABLE_X64=0`                |

See [`GitLab Runner (shell) × Nix flake check — ローカル CI 一括導入ガイド`](gitlab-runner-nix-ci-guide.md) for a deeper architectural explanation.

## 📜 License

This project is licensed under the [MIT License](LICENSE).  © 2025 Tomoyuki Kano
