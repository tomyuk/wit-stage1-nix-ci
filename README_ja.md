# wit-stage1-nix-ci ― 日本語 README

> **Weak Inhomogeneity Theory (WIT) Stage 1 実験用ローカル CI**
> **GitLab Runner（shell executor）** と **Nix flake check** だけで、Docker を使わずに Apple Silicon / Metal GPU 上で再現可能。

---

![build](https://gitlab.com/your-group/wit-stage1-nix-ci/badges/main/pipeline.svg)

このリポジトリは *Weak Inhomogeneity Theory*（WIT）プロジェクト Stage 1 の自動テストと品質保証パイプラインをホストします。

* **GPU 対応・Docker 不要** ‑ Metal / MPS / JAX‑Metal などをそのまま利用。
* **ワンコマンド検証** ‑ `nix flake check` で依存解決→ビルド→テストを一括実行。
* **CI とローカルの一致** ‑ `.gitlab-ci.yml` でも同じコマンドを呼ぶだけ。
* **完全再現性** ‑ 依存は `flake.lock` に固定。

## 📂 ディレクトリ構成

```text
.gitlab-ci.yml        # GitLab Runner パイプライン設定（shell）
flake.nix             # Nix flake ルート
flake.lock            # 依存ロックファイル
nix/
  ci.nix              # 開発シェル & テスト定義
scripts/
  run_stage1.py       # Stage 1 オーケストレーション
LICENSE               # MIT ライセンス
README.md             # 英語版
README_ja.md          # ← このファイル
```

## 🚀 クイックスタート

```bash
# 1) 必要ツールを導入（macOS 例）
curl -L https://nixos.org/nix/install | sh -- --daemon
brew install gitlab-runner direnv pre-commit

# 2) リポジトリを取得し dev シェルに入る
git clone https://gitlab.com/your-group/wit-stage1-nix-ci.git
cd wit-stage1-nix-ci
direnv allow          # nix シェルを自動ロード

# 3) テストをローカル実行
nix flake check

# 4) GitLab Runner と同じ条件で実行
gitlab-runner exec shell flake-check
```

## 🔧 開発ワークフロー

1. コードを編集（`scripts/`, `nix/` など）。
2. 必要なら `nix flake update` で依存を更新。
3. `nix flake check` が通ることを確認。
4. コミット & プッシュ。

### よく使うコマンド

| 目的                    | コマンド                                           |
| --------------------- | ---------------------------------------------- |
| 開発シェルへ入る              | `nix develop`                                  |
| 古いストア世代を削除            | `nix-collect-garbage -d`                       |
| 個別テストを実行              | `python scripts/run_stage1.py --case <name>`   |
| ローカル Runner を登録（一度だけ） | `sudo gitlab-runner register --executor shell` |

## 🐛 トラブルシューティング

| 症状                    | 対処法                                           |
| --------------------- | --------------------------------------------- |
| `nix flake check` が遅い | `nix-store --gc` & `/nix/store` キャッシュを CI に設定 |
| CI で `nix` が見つからない    | Runner の `config.toml` に Nix プロファイルを追加        |
| Metal backend がクラッシュ  | `export JAX_ENABLE_X64=0` で float32 に戻す       |

詳細は **[GitLab Runner (shell) × Nix flake check — ローカル CI 一括導入ガイド](gitlab-runner-nix-ci-guide.md)** を参照してください。

## 📜 ライセンス

本リポジトリは [MIT License](LICENSE) のもとで公開されています。© 2025 Tomoyuki Kano
