from .api.server import main,set_provider,app
from .api.file_provider import FileProvider
import logging
import os
import sys

class Args:

   results_directory = '/home/robertsone/RESULTS'
   data_directory = '/home/robertsone/TESTS'
   log_file = f'{os.getpid()}_wsgi.log'
   log_level = logging.INFO
   url = '0.0.0.0:5003'

def set_up(args):
    set_provider(
        FileProvider(
            os.path.abspath(args.data_directory),
            os.path.abspath(args.results_directory),
        )
    )
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        filename=args.log_file, filemode="w", level=args.log_level, format=log_format
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Api server starting with provider set to FileProvider")

def create_app():
   set_up(Args())
   return app

