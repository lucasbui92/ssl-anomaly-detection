
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr', 'smart_grid'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return r"D:\Program Files (x86)\OneDrive\Documents\University of Essex\Dissertation\CARLA\datasets\MSL_SMAP"
        elif database == 'ucr':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/UCR'
        elif database == 'yahoo':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/Yahoo'
        elif database == 'smd':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/SMD'
        elif database == 'swat':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/SWAT'
        elif database == 'wadi':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/WADI'
        elif database == 'kpi':
            return r"D:\Program Files (x86)\OneDrive\Documents\University of Essex\Dissertation\CARLA\datasets\kpi"
        elif database == 'swan':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/Swan'
        elif database == 'gecco':
            return '/home/zahraz/hz18_scratch/zahraz/datasets/GECCO'
        elif database == 'smart_grid':
            return r"D:\Program Files (x86)\OneDrive\Documents\University of Essex\Dissertation\CARLA\datasets\SmartGrid"
        else:
            raise NotImplementedError
