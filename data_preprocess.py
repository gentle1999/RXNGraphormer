import argparse


def main():
    try:
        from rxngraphormer.preprocess.cli import preprocess_from_config
    except ImportError as exc:
        raise SystemExit(
            "The preprocessing entrypoint requires the optional preprocessing "
            "dependencies. On compatible older hardware, install the root "
            "`preprocess` extra. On newer hardware, run preprocessing in a "
            "separate environment and reuse the generated .pt artifacts."
        ) from exc

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json', type=str, default='./config/pretrain_parameters.json')
    args = parser.parse_args()
    preprocess_from_config(args.config_json)
    
if __name__ == '__main__':
    main()
