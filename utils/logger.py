class OutFlags:
    GREEN = '\033[92m'

    INFO_VERB = '\033[97m'
    INFO = '\033[94m'
    HOPS = '\033[95m'

    END = '\033[0m'
    BOLD = '\033[1m'


def msg(tag, message, mode=OutFlags.INFO):
    if mode not in [OutFlags.HOPS, OutFlags.INFO]:
        return

    print(mode +
          OutFlags.BOLD + tag + OutFlags.END +
          OutFlags.GREEN + '::' + OutFlags.END +
          mode + message + OutFlags.END)
