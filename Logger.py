class debug_colours:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def debug(tag, message, mode=debug_colours.BLUE):
    if mode == debug_colours.BLUE:
        return
    print(mode +
          debug_colours.BOLD + tag + debug_colours.END +
          debug_colours.GREEN + '::' + debug_colours.END +
          mode + message + debug_colours.END)