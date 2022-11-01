import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="main.yaml", version_base="1.1")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.tasks.regression.trn_val_tst import process_regression
    from src.tasks.classification.trn_val_tst import process_classification
    from src.utils import utils
    import torch

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDNN VERSION:', torch.backends.cudnn.version())
        print('Number CUDA Devices:', torch.cuda.device_count())
        print('CUDA Device Name:', torch.cuda.get_device_name(0))
        print('CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1024**3)

    if config.task == "regression":
        return process_regression(config)
    elif config.task == "classification":
        return process_classification(config)
    else:
        raise ValueError(f"Unsupported task: {config.task}")


if __name__ == "__main__":
    main()
