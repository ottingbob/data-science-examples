#!/bin/bash

# Print the header
gum style \
	--foreground 212 --border-foreground 105 --border double \
	--align center --width 50 --margin "1 2" --padding "1 2" \
	'P&L 3 Statement Model Forecast' \
	'Brought to you from :tada: `yöbawb`'

# gum input \
	# --prompt.foreground 212 \
	# --cursor.foreground 105 \
  # --prompt "What rate to forecast growth at? " \
  # --placeholder "1%" \
	# --width 80 \
	# --value "2%"

echo 'What rate to format growth at?' | gum style --foreground 212 

response=$(printf "1%%\n2%%\n3%%\n4%%\nexit" | gum choose --limit 1 --cursor.foreground 212 --item.foreground 105)

if [[ "${response}" == "exit" ]]; then
	exit
fi

python financial_analyst_course_2023/p-and-l-3-statment-model.py
