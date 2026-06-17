#!/usr/bin/env bash
# Quiesce a dedicated headless inference Pi: stop clearly-unneeded background services + the desktop GUI (HDMI
# is disconnected, accessed via SSH), lower swappiness. All REVERSIBLE: `systemctl stop` (not disable) -> a reboot
# restores everything; pass "restore" to undo now. Frees CPU/memory contention + reduces scheduler jitter.
set -uo pipefail
SVCS="ollama bluetooth cups cups-browsed avahi-daemon triggerhappy colord ModemManager udisks2 rpi-eeprom-update"
if [ "${1:-}" = "restore" ]; then
  echo "restoring..."; sudo systemctl start lightdm 2>/dev/null
  for s in $SVCS; do sudo systemctl start $s 2>/dev/null; done
  echo 60 | sudo tee /proc/sys/vm/swappiness >/dev/null
  echo "restored (services started, swappiness=60, GUI back)"; exit 0
fi
echo "=== stopping desktop GUI (lightdm) ==="; sudo systemctl stop lightdm 2>/dev/null && echo "  lightdm stopped"
echo "=== stopping background services ==="
for s in $SVCS; do
  if systemctl is-active "$s" >/dev/null 2>&1; then sudo systemctl stop "$s" 2>/dev/null && echo "  stopped: $s"; fi
done
echo "=== swappiness 60 -> 10 (keep mlock'd weights resident) ==="
echo 10 | sudo tee /proc/sys/vm/swappiness >/dev/null && echo "  swappiness=$(cat /proc/sys/vm/swappiness)"
echo "=== freed: ==="; free -h | awk '/Mem|Swap/{print "  "$0}'
