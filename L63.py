
import numpy
import math

import matplotlib.pyplot
from mpl_toolkits import mplot3d

import types


def dX(t, X, a=10, b=28, c=8/3):
	"""
	Compute the vector field of L63 dynamical system.
	
	Parameters
	----------
	t : float
	X : (N, 3) array_like
		Point(s) to evaluate the L63 dynamical system vector field.
	a : float
		First parameter of the L63 dynamical system. Default is 10.
	b : float
		Second parameter of the L63 dynamical system. Default is 28.
	c : float
		Third parameter of the L63 dynamical system. Default is 8/3.
	
	Returns
	-------
	dX : (N, 3) ndarray

	"""

	return numpy.transpose(numpy.array([a*(X[:,1]-X[:,0]), X[:,0]*(b-X[:,2])-X[:,1], X[:,0]*X[:,1]-c*X[:,2]]))

def trajectory(X0, T, ΔT):
	"""
	Compute a trajectory of the (standard) L63 dynamical system.
	
	Parameters
	----------
	X0 : (N, 3) array_like
		Initial values for individual trajectories in space.
	T : int, float
		Length of the time interval.
	ΔT : float
		Time increment.	
	
	Returns
	-------
	X : (*, N, 3) ndarray
		Trajectories of the (standard) L63 dynamical system for initial values X0.
	"""

	ΔT = round(ΔT, 3)

	X = numpy.empty((math.ceil(T/ΔT)+1, X0.shape[0], 3))
	X[0,:] = X0

	X_ = X0

	for t in range(X.shape[0]-1):
		for t_ in range(round(ΔT/0.001)):
			X_ = X_ + 0.001 * dX(numpy.nan, X_)
		
		X[t+1,:] = X_
	
	return X

def rnd_trajectory(N, T, ΔT, T0=0):
	"""
	Compute a random trajectory of the (standard) L63 dynamical system.
	
	Parameters
	----------
	N : int
		Number of trajectories to compute.
	T : int, float
		Length of the time interval.
	ΔT : float
		Time increment.
	T0 : int, float
		Length of burn-in time before the start of the trajectories. Default is 0.	
	
	Returns
	-------
	X : (*, N, 3) ndarray
		Trajectories of the (standard) L63 dynamical system for initial values X0.
	"""

	X0 = 20 * numpy.random.rand(N,3) + numpy.array([-10,-10,0])

	X = trajectory(X0, T + T0, ΔT)
	X = X[math.ceil(T0/ΔT):,:,:]
	
	return X


def trajectory_plot(X, ax=None, colormap='viridis'):
	"""
	Plot trajectories of the L63 dynamical system.
	
	Parameters
	----------
	X : (*, N, 3) ndarray
		Trajectories of the L63 dynamical system.
	ax : *
		Matplotlib 3d axis to draw on. Default is None.
	colormap : str
		Default is 'viridis'.
	
	Returns
	-------
	
	"""

	# perceptually uniform colormaps: viridis, plasma, inferno, magma, cividis
	cmap = matplotlib.pyplot.get_cmap(colormap)


	if ax is None:
		ax = matplotlib.pyplot.gca(projection='3d')
	

	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.zaxis.set_ticklabels([])

	ax.xaxis.set_pane_color((1, 1, 1, 0))
	ax.yaxis.set_pane_color((1, 1, 1, 0))
	ax.zaxis.set_pane_color((1, 1, 1, 0))

	ax.xaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)
	ax.yaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)
	ax.zaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)

	N = X.shape[1]
	for n in numpy.random.permutation(N):
		ax.plot(X[:,n,0], X[:,n,1], X[:,n,2], c=cmap(n/(N-1)), linewidth=0.2)

	#return ax

def scatter_plot(X, h, ax=None, X0=None, h_range=None, colormap='viridis'):
	"""
	Plot a point cloud of the L63 dynamical system.
	
	Parameters
	----------
	X : (*, N, 3) ndarray
		Points of the L63 dynamical system.
	h : function, (N,) ndarray
	ax : *
		Matplotlib 3d axis to draw on. Default is None.
	X0 :
	h_range : 
	colormap : str
		Default is 'viridis'.	
	
	Returns
	-------
	
	"""

	# perceptually uniform colormaps: viridis, plasma, inferno, magma, cividis
	cmap = matplotlib.pyplot.get_cmap(colormap)


	if ax is None:
		ax = matplotlib.pyplot.gca(projection='3d')

	#ax = matplotlib.pyplot.axes(projection='3d')

	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.zaxis.set_ticklabels([])

	ax.xaxis.set_pane_color((1, 1, 1, 0))
	ax.yaxis.set_pane_color((1, 1, 1, 0))
	ax.zaxis.set_pane_color((1, 1, 1, 0))

	ax.xaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)
	ax.yaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)
	ax.zaxis._axinfo['grid']['color'] =  (1, 1, 1, 0)

	if type(h) == types.FunctionType:
		
		if X0 is None:
			X0 = X

		h_X0 = numpy.apply_along_axis(h, 1, X0)
	else:
		h_X0 = h

	if h_range is None:
		h_range = (min(h_X0), max(h_X0))

	h_colormap = cmap((h_X0 - min(h_range))/(max(h_range)-min(h_range)))

	
	ax.scatter(X[:,0], X[:,1], X[:,2], color=h_colormap, s=5, edgecolor=(1,1,1,0))

	#return ax
