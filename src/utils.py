from src.logger import logging
from src.exception import CustmeException
import os,sys


def save_object(file_path,obj):
    try:
        pass

    except Exception as e:
        raise CustmeException(e,sys)