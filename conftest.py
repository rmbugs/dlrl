import logging

def pytest_addoption(parser):
    parser.addoption(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing them"
    )

def pytest_configure(config):
    config.option.verbose = 2

    # Set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename='pytest.log',
        filemode='w'
    )
