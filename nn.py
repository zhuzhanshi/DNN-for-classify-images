#!/usr/bin/env python
# coding:utf8

import numpy as np


def data():
	t = np.random.rand(3,2)
	print(t)
	print(t.all()<1)



if __name__ == '__main__':
	data()