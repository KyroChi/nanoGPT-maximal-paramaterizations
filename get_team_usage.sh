#!/bin/bash

QOS_LIST=("wm" "k2m" "iq" "ad" "hao" "research" "lowprio" "main")

# Header for per-user breakdown
printf "%-10s %-15s %-10s\n" "Team" "USER" "NODES"

for qos in "${QOS_LIST[@]}"; do
  /usr/bin/squeue --noheader --Format=username,qos,numnodes,nodelist -t R | sort | \
  awk -v qos="$qos" '
    function already_seen(user, node) {
      return (user "|" node) in seen
    }

    $2 == qos {
      if ($3 == 1) {
        # Handle single-node job
        split($4, nodes, /,/)
        for (i in nodes) {
          node = nodes[i]
          if (!already_seen($1, node)) {
            seen[$1 "|" node] = 1
            user[$1]++
            total++
          }
        }
      } else {
        # Multi-node jobs – count numnodes as-is
        user[$1] += $3
        total += $3
      }
    }

    END {
      if (length(user) == 0)
        printf "%-10s %-15s %-10s\n", qos, "-", "0"
      else {
        for (u in user)
          printf "%-10s %-15s %-10d\n", qos, u, user[u]
      }
      printf "TOTAL:%s:%d\n", qos, total > "/tmp/team_total_" qos
    }'
done

# Blank line before totals
echo ""
printf "%-10s %-15s %-10s\n" "Team" "USER" "NODES"

for qos in "${QOS_LIST[@]}"; do
  if [[ -f /tmp/team_total_$qos ]]; then
    total=$(cut -d':' -f3 /tmp/team_total_$qos)
    printf "%-10s %-15s %-10d\n" "$qos" "TOTAL" "$total"
    rm -f /tmp/team_total_$qos
  else
    printf "%-10s %-15s %-10d\n" "$qos" "TOTAL" 0
  fi
done