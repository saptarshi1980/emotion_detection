from logger import logging
from exception import CustomException
import os,sys

if __name__=='__main__':
    try:
        10/0
    except Exception as e:
        
        raise CustomException(e,sys)
        