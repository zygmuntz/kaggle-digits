"""
general csv dataset wrapper for pylearn2, here with one hot and scaling/255 for digits
"""

import csv
import numpy as np
import os

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

class DigitsDataset( DenseDesignMatrix ):

	def __init__(self, 
			path = 'train.csv',
			one_hot = False,
			labels = True,
			headers = True):

		self.path = path
		self.one_hot = one_hot
		# re-map those
		self.expect_labels = labels
		self.expect_headers = headers
		
		self.no_classes = 10
		self.view_converter = None

		# and go

		self.path = preprocess( self.path )
		X, y = self._load_data()
		X = X / 255.
		
		super( CSVDataset, self ).__init__( X=X, y=y )

	def _load_data( self ):
	
		assert self.path.endswith('.csv')
	
		if self.expect_headers:
			data = np.loadtxt( self.path, delimiter = ',', dtype = 'int', skiprows = 1 )
		else:
			data = np.loadtxt( self.path, delimiter = ',', dtype = 'int' )
		
		if self.expect_labels:
			y = data[:,0]
			X = data[:,1:]

			if self.one_hot:
				one_hot = np.zeros(( y.shape[0], self.no_classes ), dtype='float32' )
				for i in xrange( y.shape[0] ):
					one_hot[i,y[i]] = 1.
				y = one_hot

		else:
			X = data
			y = None

		return X, y
