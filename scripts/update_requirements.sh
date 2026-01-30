#!/bin/bash

PROJECT_ROOT="/workspaces/pvenergy"
IN_FILE="$PROJECT_ROOT/requirements.in"
TXT_FILE="$PROJECT_ROOT/requirements.txt"
TEMP_IN="$PROJECT_ROOT/requirements.tmp"

pipreqs "$PROJECT_ROOT/src" --savepath "$TEMP_IN" --force --mode no-pin 2>/dev/null

while read -r line; do
    pkg_name=$(echo "$line" | sed 's/[<>=].*//' | tr -d '[:space:]')
    if [ ! -z "$pkg_name" ]; then
        if ! grep -qi "^${pkg_name}[<>= ]\|^${pkg_name}$" "$IN_FILE"; then
            echo "$pkg_name" >> "$IN_FILE"
            echo "Added package to requirements.in: $pkg_name"
        fi
    fi
done < "$TEMP_IN"

rm -f "$TEMP_IN"

sort -u -f -o "$IN_FILE" "$IN_FILE"

if pip-compile "$IN_FILE" \
    --output-file="$TXT_FILE" \
    --resolver=backtracking \
    --strip-extras \
    --no-annotate; then

    INSTALL_LIST=$(pip install -r "$TXT_FILE" --dry-run --report - 2>/dev/null | grep -oP '(?<="name": ")[^"]*' || echo "")

    if [ -n "$INSTALL_LIST" ]; then
        echo "The following packages need to be installed or updated:"
        echo "$INSTALL_LIST"
        echo
        read -p "Start installation? (y/N): " confirm
        if [[ "$confirm" == [yY] ]]; then
            pip install -r "$TXT_FILE"
        fi
    else
        echo "Everything is up to date."
    fi
else
    echo "Error while compiling."
    exit 1
fi