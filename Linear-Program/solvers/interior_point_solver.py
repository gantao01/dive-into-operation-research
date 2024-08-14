import numpy as np
import sympy
from numpy.linalg import matrix_rank


class InteriorPointSolver:
	""" This class implements primal-dual (infeasible) interior-point method to solve LPs """

	def solve(self, c, A, b, epsilon=0.0001):
		"""
		This method solves the std form LP min (c.T * x) s.t. Ax = b, x >= 0 using primual-dual (infeasible) interior-point method.

		Parameters:
			c, A, b (np arrays): specify the LP in standard form
			epsilon		(float): duality gap threshold, specifies termination criteria

		Returns:
			x 		(np array): solution to the LP
		"""

		# ensure dimensions are okay
		assert A.shape[0] == b.shape[0], 'first dims of A and b must match, check input!'
		assert A.shape[1] == c.shape[0], 'second dim of A must match first dim of c, check input!'

		# ensure A is full rank, drop redundant rows if not
		if matrix_rank(A) < min(A.shape[0], A.shape[1]):
			print('A is not full rank, dropping redundant rows')
			_, pivots = sympy.Matrix(A).T.rref()
			A = A[list(pivots)]
			print('Shape of A after dropping redundant rows is {}'.format(A.shape))

		m = A.shape[0]
		n = A.shape[1]

		# initial solution (x_0, lambda_0, s_0) > 0 [lambda is variable l in code]
		# note that this is not a feasible solution in general
		# but it should tend towards feasibility by itself with iterations
		# therefore initially duality gap might show negative
		# since this is the infeasible-interior-point algorithm
		x = np.ones(shape=(n, ))
		l = np.ones(shape=(m, ))
		s = np.ones(shape=(n, ))

		# set iteration counter to 0 and mu_0
		k = 0

		# main loop body
		while abs(np.dot(x, s)) > epsilon:

			# print iteration number and progress
			k += 1
			primal_obj = np.dot(c, x)
			dual_obj = np.dot(b, l)
			print('iteration #{}; primal_obj = {:.5f}, dual_obj = {:.5f}; duality_gap = {:.5f}'.format(k, primal_obj, dual_obj, primal_obj - dual_obj))

			# choose sigma_k and calculate mu_k
			sigma_k = 0.4
			mu_k = np.dot(x, s) / n

			# create linear system A_ * delta = b_
			A_ = np.zeros(shape=(m + n + n, n + m + n))
			A_[0:m, 0:n] = np.copy(A)
			A_[m:m + n, n:n + m] = np.copy(A.T)
			A_[m:m + n, n + m:n + m + n] = np.eye(n)
			A_[m + n:m + n + n, 0:n] = np.copy(np.diag(s))
			A_[m + n:m + n + n, n + m:n + m + n] = np.copy(np.diag(x))

			b_ = np.zeros(shape=(n + m + n, ))
			b_[0:m] = np.copy(b - np.dot(A, x))
			b_[m:m + n] = np.copy(c - np.dot(A.T, l) - s)
			b_[m + n:m + n + n] = np.copy( sigma_k * mu_k * np.ones(shape=(n, )) - np.dot(np.dot(np.diag(x), np.diag(s)), np.ones(shape=(n, ))) )

			# solve for delta
			delta = np.linalg.solve(A_, b_)
			delta_x = delta[0:n]
			delta_l = delta[n:n + m]
			delta_s = delta[n + m:n + m + n]

			# find step-length alpha_k
			alpha_max = 1.0
			for i in range(n):
				if delta_x[i] < 0:
					alpha_max = min(alpha_max, -x[i]/delta_x[i])
				if delta_s[i] < 0:
					alpha_max = min(alpha_max, -s[i]/delta_s[i])
			eta_k = 0.99
			alpha_k = min(1.0, eta_k * alpha_max)

			# create new iterate
			x = x + alpha_k * delta_x
			l = l + alpha_k * delta_l
			s = s + alpha_k * delta_s

		# print difference between Ax and b
		diff = np.dot(A, x) - b
		print('Ax - b = {}; ideally it should have been zero vector'.format(diff))
		print('norm of Ax - b is = {}; ideally it should have been zero'.format(np.linalg.norm(diff)))

		return x