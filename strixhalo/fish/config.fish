source /usr/share/cachyos-fish-config/cachyos-config.fish

# overwrite greeting
# potentially disabling fastfetch
#function fish_greeting
#    # smth smth
#end
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

# dotfiles bare-repo alias (strixhalo-box backups)
alias dotfiles="git --git-dir=$HOME/.dotfiles --work-tree=$HOME"
