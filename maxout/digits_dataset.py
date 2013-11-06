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
			expect_labels = True,
			expect_headers = True):

		self.path = path
		self.one_hot = one_hot
		self.expect_labels = expect_labels
		self.expect_headers = expect_headers
		
		self.view_converter = None

		# and go

		self.path = preprocess( self.path )
		X, y = self._load_data()
		X = X / 255.
		
		super( DigitsDataset, self ).__init__( X=X, y=y )

	def _load_data( self ):
	
		assert self.path.endswith('.csv')
	
		if self.expect_headers:
			data = np.loadtxt( self.path, delimiter = ',', dtype = 'int', skiprows = 1 )
		else:
			data = np.loadtxt( self.path, delimiter = ',', dtype = 'int' )
		
		if self.expect_labels:
			y = data[:,0]
			X = data[:,1:]
			
			# get unique labels and map them to one-hot positions
			labels = np.unique(y)
			labels = { x: i for i, x in enumerate( labels ) }

			if self.one_hot:
				one_hot = np.zeros(( y.shape[0], len( labels )), dtype='float32' )
				for i in xrange( y.shape[0] ):
					label = y[i]
					label_position = labels[label]
					one_hot[i,label_position] = 1.
				y = one_hot
		else:
			X = data
			y = None

		return X, y
