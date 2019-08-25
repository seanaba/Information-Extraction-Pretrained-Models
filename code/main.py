"""
Information extraction (pre-trained models)
1- Collect  documents
2- Extract information
"""
import os
from extract import ExtractDoc


if __name__ == "__main__":
    """
    Providing data path 
    """
    code_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(code_path)
    data_path = os.path.join(base_path, 'data')
    try:
        info_extract_obj = ExtractDoc(data_path)
        info_extract_obj.extract_info()
    finally:
        del info_extract_obj
