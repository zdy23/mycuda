#!/usr/bin/env bash
# 混合 swap 方案: zram(压缩,优先) + 小磁盘 swap(2G 安全网)
# 用法: bash setup_zram_hybrid.sh   (中途会提示输入 sudo 密码)
set -euo pipefail

echo "==> [1/5] 安装 zram-tools"
sudo apt-get update -qq
sudo apt-get install -y zram-tools

echo "==> [2/5] 配置 zram: zstd 压缩, 大小为内存的 25% (~3.75G), 优先级 100"
sudo tee /etc/default/zramswap >/dev/null <<'EOF'
# zram 配置 (混合方案: zram 优先, 磁盘 swap 兜底)
ALGO=zstd
PERCENT=25
PRIORITY=100
EOF

echo "==> [3/5] 启用 zram (restart: 包安装时 postinst 可能已用默认配置启动过, 必须 restart 重新读配置)"
sudo systemctl enable zramswap.service
sudo systemctl restart zramswap.service

echo "==> [4/5] 把 /swap.img 从 4G 缩小到 2G (兜底安全网)"
sudo swapoff /swap.img
sudo truncate -s 2G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img >/dev/null
sudo swapon /swap.img

echo "==> [5/5] 验证"
swapon --show
zramctl

echo
echo "完成: zram(压缩,优先) + /swap.img 2G(兜底)。fstab 无需改动, 重启后配置保持。"
echo "回滚: sudo systemctl disable --now zramswap && sudo apt purge -y zram-tools"
