import argparse


args = None


def parse_arguments():
    """Parse command line arguments for federated learning."""
    parser = argparse.ArgumentParser(description="Federated Learning with Knowledge Distillation")
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', help="Dataset name")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument("--non_iid_degree", type=float, default=1.0, help="Non-IID degree for Dirichlet distribution")
    
    # Training settings
    parser.add_argument('--trainer_num', type=int, default=8, help='Number of trainers')
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--bs', type=int, default=64, help="Batch size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=128, help="Batch size for testing")
    parser.add_argument('--round', type=int, default=200, help="Communication rounds")
        
    # Optional config file
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    if args.config is not None:
        get_config(args)

    return args


def get_config(args):
    """Load configuration from file."""
    load_args = {}
    with open(args.config, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            load_args[key] = value
    args.__dict__.update(load_args)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
