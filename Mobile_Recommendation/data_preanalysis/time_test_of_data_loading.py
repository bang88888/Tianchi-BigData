# -*- coding: utf-8 -*
    
'''
@author: kuaidouai
'''

import os
import sys
import timeit
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../../data'))

start_time = timeit.default_timer()

# data loading using pandas
with open(os.path.join(data_dir, "fresh_comp_offline", "tianchi_fresh_comp_train_user.csv"), mode = 'r') as data_file:
    df = pd.read_csv(data_file)
    
end_time = timeit.default_timer()

print(('The code for file ' + os.path.split(__file__)[1] +
       ' ran for %.2fm' % ((end_time - start_time) / 60.)), file = sys.stderr)
