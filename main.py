"""
The main file to run the program.
Please edit the config file in config/config.yaml and run the code.
Made by sayednadim (sayednadim.github.io).
Use it as you wish.
cheers
"""

from src.utils.config import get_config
from src.evaluation_metrics.evaluation import single_evaluation, multiple_evaluation


def main(config):
    """
    Runs the metrics evaluation operations
    and prints the results in the terminal.
    If the save option is set true in the
    config file, then the results will be
    saved in the results directory as a csv
    file.

    Supports both single_evaluation or multiple_evaluation

    args:
    config - path of the config file.
    returns - None.
    """
    if config['multiple_evaluation']:  # check for multiple evaluation
        multiple_evaluation(config)

    else:
        single_evaluation(config)


if __name__ == '__main__':
    print(
        "#====================================================================================================#"
        "\n"
        "#                                Common Image Evaluation Metrics                                     #"
        "\n"
        "#====================================================================================================#"
        "\n"
        "# Implementation of Common Image Evaluation Metrics by Sayed Nadim (sayednadim.github.io)            #"
        "\n"
        "# The repo is built based on full reference image quality metrics such as L1, L2, PSNR, SSIM, LPIPS. #"
        "\n"
        "# and feature-level quality metrics such as FID, IS. It can be used for evaluating image denoising,  #"
        "\n"
        "# colorization, inpainting, deraining, dehazing etc. supervised tasks.                               #"
        "\n"
        "#====================================================================================================#"
    )
    CONFIG_PATH = 'src/config/config.yaml'
    CONFIG_FILE = get_config(CONFIG_PATH)
    main(CONFIG_FILE)
