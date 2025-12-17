
import os
from pathlib import Path

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'smart_grid'}
        assert(database in db_names)

        if database == 'smart_grid':
            BASE_DIR = Path(__file__).resolve().parent.parent
            return BASE_DIR / "datasets" / "SmartGrid"
        else:
            raise NotImplementedError
