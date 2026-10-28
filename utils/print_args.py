def print_args(args):
    for name, value in sorted(vars(args).items()):
        print(f"{name}: {value}")
