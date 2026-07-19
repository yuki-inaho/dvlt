# 環境セットアップ・オンボーディングガイド

**作成日**: 2026-07-19
**対象**: 新しいセッション、開発メンバー、VGGT-Omega backend 利用者
**プロジェクト**: Déjà View: Looping Transformers for Multi-View 3D Reconstruction（[yuki-inaho/dvlt](https://github.com/yuki-inaho/dvlt)）
**目的**: Pixi を使い、DVLT と VGGT-Omega 評価バックエンドの GPU 実行環境を再現可能にする。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**Déjà View（DVLT）**
順不同の複数画像からカメラ姿勢、深度、3D 点群を推定する、再帰型 Transformer ベースの multi-view 3D reconstruction 実装です。

### 最終目標

DVLT の学習・評価・可視化を再現可能に実行し、必要に応じて VGGT、VGGT-Omega、Depth-Anything-3、MapAnything、Pi3 の評価ラッパーを利用します。

本ガイドの対象は **DVLT + VGGT-Omega** の GPU 実行環境です。VGGT-Omega は評価専用で、DVLT 内から学習することはできません。

### 主要コンポーネント

* `src/dvlt/model/dvlt/` — DVLT の本体モデル
* `src/dvlt/model/vggt_omega/` — 上流 `vggt_omega` を DVLT 標準出力へ変換する評価ラッパー
* `src/dvlt/scripts/test.py` — Hydra 設定を使う評価 CLI
* `src/dvlt/scripts/gradio_demo.py` — 対話／オフラインの推論・可視化 CLI
* `pixi.toml` / `pixi.lock` — Python、CUDA、PyTorch、実行タスクを固定する環境定義

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類 | 状態 | 説明 |
| --- | --- | --- |
| Pixi 環境定義 | 🟢 | Python 3.11、PyTorch 2.5.1、CUDA 12.4 を `pixi.toml` と `pixi.lock` に固定済み |
| VGGT-Omega backend | 🟢 | 公式 upstream のコミット `39a0cb8af88554f15ddcb5354cd52bde588fa014` を固定して導入するタスクを定義済み |
| GPU smoke test | 🟢 | checkpoint の strict load、2-view CUDA forward、DVLT のカメラ／深度／3D 点群後処理を検証するスクリプトを追加済み |
| 基本設定テスト | 🟢 | `tests/config/test_schema.py` の 2 件が成功。未導入の任意 baseline は warning として skip |
| 学習・全 benchmark | ⚪ | データセットと任意 backend を揃えた上で別途実行 |

### 依存パッケージのインストール状態

新しい clone や新しいマシンでは環境がありません。**必ず** セクション 4 の `pixi install --locked` を実行してください。

Pixi の `setup`／`smoke` タスクは DVLT を editable install し、固定済み VGGT-Omega を `--no-deps` で導入します。上流 VGGT-Omega の `numpy<2` 宣言と DVLT の `numpy==2.4.4` は競合するため、DVLT の固定値を優先しています。実 checkpoint 推論で互換性を確認済みなので、独自判断で NumPy を変更しないでください。

### 未実装・これから着手する項目

* training 用データセットの準備と `dataverse` backend の導入
* 必要に応じた VGGT、Depth-Anything-3、MapAnything、Pi3 backend の導入
* benchmark データを使う精度評価
* Gradio demo 用の `demos` extra の導入

### 重要なファイル／ディレクトリ

```text
dvlt/
├── README.md                         # プロジェクト全体の概要と推論例
├── pyproject.toml                    # DVLT の Python パッケージ定義
├── pixi.toml                         # Pixi 環境・タスク定義
├── pixi.lock                         # 解決済み依存バージョン
├── docs/
│   ├── INSTALL.md                    # upstream による一般的な導入手順
│   ├── ONBOARDING.md                 # 本ドキュメント
│   └── TESTING.md                    # テストカテゴリの説明
├── scripts/
│   └── smoke_vggt_omega.py           # strict checkpoint load + CUDA 推論 smoke
├── src/dvlt/model/vggt_omega/        # VGGT-Omega の DVLT wrapper
└── tests/                            # unit・integration テスト
```

checkpoint は Git 管理しません。smoke test は次の順で checkpoint を探します。

1. `VGGT_OMEGA_CHECKPOINT` 環境変数
2. `dvlt/checkpoints/vggt_omega_1b_512.pt`
3. `dvlt/../checkpoints/vggt_omega_1b_512.pt`

---

## 3. 前提条件の確認

### 3.1 システム情報の確認

```bash
cat /etc/os-release | grep -E '^(NAME|VERSION)='
uname -r
nproc
free -h
pwd
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

GPU smoke test には NVIDIA GPU と CUDA 12.4 runtime を利用できるドライバが必要です。2026-07-19 の検証環境は RTX 4090（24GB）でした。

### 3.2 必須ツールの存在確認

```bash
git --version
~/.pixi/bin/pixi --version
~/.pixi/bin/pixi task list
```

通常のコマンドとして Pixi を使う場合は、セッション中に次を実行します。

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

### 3.3 Git ブランチ・コミット確認

```bash
git branch --show-current
git log --oneline -1
git status --short --branch
```

---

## 4. 環境セットアップ手順

### 4.1 リポジトリと Pixi の準備

```bash
git clone https://github.com/yuki-inaho/dvlt.git
cd dvlt

# Pixi がない場合だけ実行する
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### 4.2 ロックファイルから環境を作成（最重要）

```bash
pixi install --locked
```

このコマンドは `.pixi/` に環境を作成します。CUDA/PyTorch/Open3D を含むため、初回は数 GB のダウンロードが発生します。

主要な固定バージョン:

* Python 3.11
* PyTorch 2.5.1 / torchvision 0.20.1
* CUDA runtime 12.4
* NumPy 2.4.4
* VGGT-Omega upstream `39a0cb8af88554f15ddcb5354cd52bde588fa014`

### 4.3 DVLT と VGGT-Omega backend を導入

```bash
pixi run setup
```

このタスクは DVLT を editable install し、VGGT-Omega を以下の固定 commit から導入します。

```bash
python -m pip install --no-deps -e .
python -m pip install --no-deps \
  git+https://github.com/facebookresearch/vggt-omega.git@39a0cb8af88554f15ddcb5354cd52bde588fa014
```

### 4.4 VGGT-Omega checkpoint を準備

`vggt_omega_1b_512.pt` を、前節の探索先のいずれかに置きます。共有 checkpoint ディレクトリを親へ置く例は以下です。

```bash
mkdir -p ../checkpoints
# ダウンロード済みの vggt_omega_1b_512.pt を ../checkpoints/ に配置する
export VGGT_OMEGA_CHECKPOINT="$PWD/../checkpoints/vggt_omega_1b_512.pt"
```

checkpoint とコードにはそれぞれライセンスが適用されます。利用前に [VGGT-Omega のライセンス](https://github.com/facebookresearch/vggt-omega/blob/main/LICENSE) と DVLT の `LICENSES/` を確認してください。

---

## 5. 動作確認

### 5.1 環境・backend import の確認

```bash
pixi run smoke-import
```

期待値には PyTorch `2.5.1`、CUDA runtime `12.4`、`cuda_available: True`、GPU 名、`VGGTOmega` wrapper/backend 名が含まれます。

### 5.2 実 checkpoint による GPU smoke test

```bash
pixi run smoke-vggt-omega
# import 確認も含めて一括実行する場合
pixi run smoke
```

この smoke は 64×64 のランダムな 2-view 入力を使います。checkpoint を strict load し、CUDA forward 後に DVLT の標準出力へ変換して、深度 `(1, 2, 64, 64)` と world point `(1, 2, 64, 64, 3)` の shape・有限値を検証します。精度評価ではなく、導入経路の動作確認です。

### 5.3 設定スキーマの確認

```bash
pixi run pytest -q tests/config/test_schema.py
```

期待値は `2 passed` です。VGGT、Depth-Anything-3、MapAnything、Pi3 を導入していない場合の skip warning は想定内です。

### 5.4 利用可能なタスクの確認

```bash
pixi task list
```

主なタスクは `setup`、`smoke-import`、`smoke-vggt-omega`、`smoke` です。

---

## 6. トラブルシューティング

### 問題1: `pixi: command not found`

Pixi の標準インストール先が `~/.pixi/bin` で、PATH に入っていない状態です。

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

### 問題2: `pixi install --locked` が失敗する

古い `pixi.lock` と異なる manifest を使っている、または clone が不完全な可能性があります。

```bash
git status --short --branch
git pull --ff-only
pixi install --locked
```

ロックファイルを勝手に更新せず、まず現在ブランチの manifest と lockfile を揃えてください。

### 問題3: `torch.cuda.is_available()` が `False`

```bash
nvidia-smi
pixi run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

GPU がない環境では `smoke-vggt-omega` は実行できません。CPU で設定だけを確認する場合は `pixi run smoke-import` を使ってください。

### 問題4: VGGT-Omega checkpoint が見つからない

```bash
ls -lh ../checkpoints/vggt_omega_1b_512.pt
export VGGT_OMEGA_CHECKPOINT="/absolute/path/to/vggt_omega_1b_512.pt"
pixi run smoke-vggt-omega
```

checkpoint は Git の ignore 対象です。リポジトリへ追加・commit しないでください。

### 問題5: NumPy の制約競合を報告される

上流 VGGT-Omega は `numpy<2`、DVLT は `numpy==2.4.4` を指定しています。この環境では DVLT の固定値を優先し、VGGT-Omega を `--no-deps` で導入します。`pixi.toml` の NumPy 固定を変更せず、まず `pixi run smoke` の結果で判断してください。

---

## 7. 次のステップ

1. checkpoint と入力画像を用意し、`python -m dvlt.scripts.gradio_demo --offline` で推論結果を出力する。
2. benchmark を使う場合は `docs/data/DATA.md` を読み、対象データセットの前処理を行う。
3. 学習を行う場合は `docs/INSTALL.md` の `dataverse` 手順と `src/dvlt/config/experiments/user/` のデータルート設定を完了する。
4. 他 baseline が必要な場合は `docs/INSTALL.md` の該当 upstream package を追加する。

---

## 8. 環境セットアップ完了チェックリスト

* [ ] `nvidia-smi` で期待する GPU とドライバを確認した
* [ ] `pixi --version` と `pixi task list` が成功した
* [ ] 想定する Git ブランチ・最新コミットで作業している
* [ ] `pixi install --locked` が成功した
* [ ] `pixi run setup` が成功した
* [ ] VGGT-Omega checkpoint を準備した、または `VGGT_OMEGA_CHECKPOINT` を設定した
* [ ] `pixi run smoke-import` で CUDA と backend import を確認した
* [ ] `pixi run smoke` が成功した
* [ ] `pixi run pytest -q tests/config/test_schema.py` が成功した
* [ ] `README.md`、`docs/INSTALL.md`、`docs/TESTING.md` を読んだ

---

## 9. 更新履歴

* 2026-07-19 UTC: 初版作成。Pixi による DVLT + VGGT-Omega 環境、checkpoint smoke、再現手順、トラブルシューティングを記録。

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが、短時間で同一の開発・実行環境を再現し、既存タスクを引き継げるようにすることを目的としています。

問題が発生した場合は、本ドキュメントに加えて `README.md`、`docs/INSTALL.md`、`docs/TESTING.md`、Git のコミット履歴を参照してください。
