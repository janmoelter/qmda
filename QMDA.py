
import math
import numpy

import scipy.linalg
import scipy.spatial.distance

import time

def kernel_bandwidth_tuning(Δ):
	"""
	Perform automatic parameter tuning of a Gaussian kernel.
	
	References: Berry, Giannakis, and Harlim, Phys. Rev. E 91:032915 (2015)
	
	Parameters
	----------
	Δ : (N, N) array_like
		Distance matrix of the Gaussian kernel.
	
	Returns
	-------
	ε : float
		Optimal bandwidth.
	d : float
		Dimensionality of the data manifold.
	"""

	#logT = lambda ε : numpy.log( numpy.sum(numpy.exp(- Δ[:,:,numpy.newaxis] / ε), axis=(0,1)) / N**2 )
	#logT = lambda ε : numpy.log( numpy.sum(numpy.exp(- Δ[:,:,numpy.newaxis] / ε), axis=(0,1)) )
	logT = lambda ε : numpy.log( numpy.sum(numpy.exp(- Δ / ε), axis=(0,1)) )

	h = 1e-10

	ε = 2**np.arange(-30, 10 + 0.1, 0.1)
	#
	# dlogT_dlogε = (logT(ε + h) - logT(ε)) / (numpy.log(ε + h) - numpy.log(ε))
	# \_ Although elegant, this produces very large matrices and turns out to be rather slow.
	#
	dlogT_dlogε = numpy.array([ logT(e + h) - logT(e) for e in ε ]) / (numpy.log(ε + h) - numpy.log(ε))


	#plt.semilogx(ε,dlogT_dlogε);
	#plt.title('$\\frac{\\mathrm{d} ln T}{\\mathrm{d} ln ε}$');


	i_max = numpy.argmax(dlogT_dlogε)
	return ε[i_max] , 2 * dlogT_dlogε[i_max]


def variable_bandwidth_kernel(Y, metric='sqeuclidean'):
	"""
	Compute a variable-bandwidth Gaussian kernel.
	
	References: Giannakis, Das, and Slawinska, arXiv:1808.01515 (2018)
				Berry, Giannakis, and Harlim, Phys. Rev. E 91:032915 (2015)
	
	Parameters
	----------
	Y : (N, m) array_like
		Time-ordered sequence of data points in m dimenional feature space.
	metric : str or function, optional
		The distance metric to use. This distance will not be squared. Default is 'sqeuclidean'.
	
	Returns
	-------
	K : (N, N) ndarray
		Variable-bandwidth Gaussian kernel matrix.
	"""

	π = math.pi
	
	N, _ = Y.shape

	#τ = time.time()
	#print('scipy.spatial.distance.pdist ', end='')
	Δ = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Y, metric=metric))
	#print('[{}]'.format(time.time() - τ))
	#τ = time.time()
	#print('kernel_bandwidth_tuning ', end='')
	ε, m = kernel_bandwidth_tuning(Δ)
	#print('[{}]'.format(time.time() - τ))
	#print('(ε, d) =', (ε, m))

	inv_σ = (1 / numpy.sqrt(π * ε)) * (numpy.sum(numpy.exp(- Δ / ε), axis=1) / N)**(1/m)

	Δ = Δ * numpy.outer(inv_σ, inv_σ)

	#τ = time.time()
	#print('kernel_bandwidth_tuning ', end='')
	ε, d = kernel_bandwidth_tuning(Δ)
	#print('[{}]'.format(time.time() - τ))
	#print('(ε, d) =', (ε, d))


	return numpy.exp( - Δ / ε )


def data_driven_basis(Y, L=1):
	"""
	Compute the data-driven basis.
	
	References: Das, Giannakis, and Slawinska, arXiv:1808.01515 (2018)
	
	Parameters
	----------
	Y : (N, m) array_like
		Time-ordered sequence of data points in m dimenional feature space.
	L : int
		The number of largest basis vectors to compute. Default is 1.
	
	Returns
	-------
	λ : (L,) ndarray
		Eigenvalues.
	φ : (N, L) ndarray
		Left-singular vectors of the kernel matrix; the data-driven basis vectors.
	γ : (N, L) ndarray
		Right-singular vectors of the kernel matrix.
	"""

	N, _ = Y.shape

	K = variable_bandwidth_kernel(Y) / N
	d = numpy.sum(K, axis=1)
	q = numpy.dot(K, 1/d)

	K = numpy.diag(1/d) @ K @ numpy.diag(numpy.sqrt(1/q))

	τ = time.time()
	print('scipy.linalg.svd ', end='')
	U, s, Vh = scipy.linalg.svd(K)
	print('[{}]'.format(time.time() - τ))
	V = Vh.conj().T

	λ = s[:L]**2

	φ = U[:,:L]
	# scipy.linalg.norm(φ, axis=0) = array([1., 1., ... 1.])
	φ = math.sqrt(N) * φ

	γ = V[:,:L]
	# scipy.linalg.norm(γ, axis=0) = array([1., 1., ... 1.])
	γ = math.sqrt(N) * γ

	return φ, λ, γ

def koopman_operator(φ, shift_Q=[1]):
	"""
	Compute the data-driven matrix representations of the Koopman operators.
	
	References: Giannakis, Phys. Rev. E 100:032207 (2019)
	
	Parameters
	----------
	φ : (N, L) ndarray
		The data-driven basis vectors.
	shift_Q : list
		Default is [1].
	
	Returns
	-------
	U : (*, L, L) ndarray
		Matrix representations of the Koopman operators for different shift values.
	V : (L, L) ndarray
		Matrix representation of the generator of the Koopman operators.
	"""

	N, _ = φ.shape

	#U = lambda q: numpy.transpose(φ[:-q,:]) @ φ[q:,:] / N
	U = lambda q: numpy.transpose(numpy.conjugate(φ[:N-q,:])) @ φ[q:,:] / N
	U_q = numpy.array([U(q) for q in shift_Q])

	#V = 1j * scipy.linalg.logm(U(1), disp=False)

	return U_q

def conditional_averaging(h, S=10):
	"""
	Compute a discretisication of a function performing conditional averaging.

	References: Giannakis, Phys. Rev. E 100:032207 (2019)

	Parameters
	----------
	h : (N,) ndarray
		Function evaluated along a trajectory of points.
	S : int
		Discretisation parameter. Default is 10.	

	Returns
	-------
	h : (N,) ndarray
		Discretised function.
	a : (S,) ndarray
		Discretised range of the function.
	Ξ : (S+1,) ndarray
		Partition of the function's range with the first and last element being -∞ and +∞, respectively.
	iM : (S,) list
		Indices of function values within each element of the partition of the function's range.
	"""

	J = numpy.linspace(0, 1, S+1)

	Ξ = numpy.quantile(h, q=J, interpolation='linear')
	Ξ[0], Ξ[-1] = -numpy.inf, +numpy.inf


	iM = [ numpy.where(numpy.logical_and(Ξ[i] <= h, h < Ξ[i+1])) for i in range(S) ]


	a = numpy.array([ numpy.mean(h[iM[i]]) for i in range(S) ])


	h = numpy.zeros(h.shape)
	for i in range(S):
		h[iM[i]] = a[i]


	return h, a, Ξ, iM

def observable_spectral_projectors(h, φ, S=10):
	"""
	Compute the data-driven matrix representations of the spectral projectors associated with an observable.

	References: Giannakis, Phys. Rev. E 100:032207 (2019)
	
	Parameters
	----------
	h : (N,) ndarray
		Function evaluated along a trajectory of points.
	φ : (N, L) ndarray
		The data-driven basis vectors.
	S : int
		Discretisation parameter. Default is 10.	
	
	Returns
	-------
	π : function
		Affiliation function that maps an observed value to the index of a spectral projector.
	E : (S, L, L) ndarray
		Matrix representations of the spectral projectors corresponding to the discretised observable.
	"""

	_, a, Ξ, iM = conditional_averaging(h, S=S)

	π = lambda a: numpy.digitize(a, Ξ[1:-1])


	N, L = φ.shape

	E = numpy.empty((S, L, L))

	for i in range(S):
		I = numpy.zeros(h.shape)
		I[iM[i]] = 1

		E[i,:,:] = ( numpy.transpose(φ) @ numpy.diag(I) @ φ ) / N

	return a, π, E
