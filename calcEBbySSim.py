import numpy as np 
import sys 
import itertools
from math import prod
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Usage: python calcEBbySSIM.py [float/double] [ori_data_file] [ssim_target] [number_of_dims] [dims (slowest last)] [block_size (-1 for global)] [shift_size (optional, for blockwise)] [output_file (optional, for blockwise)]")
		print("Example 1: python calcEBbySSIM.pyfloat pressure.f32 0.999 3 256 384 384 -1")
		print("Example 2: python calcEBbySSIM.py float pressure.f32 0.999 3 256 384 384 8 4 pressure_blockwise_ssim.dat")
		sys.exit(1)

	datatype = np.double if sys.argv[1] == "double" else np.float32 
	ori_data_path = sys.argv[2]
	ssim_target = float(sys.argv[3]) 
	num_of_dims = int(sys.argv[4])
	cur_argc = 5
	dims = tuple( (int(sys.argv[cur_argc + i]) for i in range(num_of_dims) ) ) 
	cur_argc += num_of_dims
	block_size = int(sys.argv[cur_argc])
	cur_argc += 1 
	is_global = block_size < 0
	write_to_file = False
	if not is_global:
		shift = int(sys.argv[cur_argc])
		cur_argc += 1 
		if len(sys.argv) > cur_argc:
			write_to_file = True 
			output_path = sys.argv[cur_argc]
			cur_argc += 1 

	ori_data = np.fromfile(ori_data_path, dtype = datatype).reshape(dims)
	#ori_rng = np.max(ori_data) - np.min(ori_data)
	#abs_eb = ori_rng * rel_eb

	a = 2.0 
	b = 2.0 
	def calc_stats_single(dat):
		mx = np.max(dat)
		mi = np.min(dat)
		mean = np.mean(dat)
		var = np.var(dat)
		std = np.sqrt(var)
		return mx, mi, mean, var, std

	def calc_eb_by_ssim(ori, ssim_target, a = 2.0, b = 2.0, rng_ori = None):
		K1 = 0.01
		K2 = 0.03
		if rng_ori == None:
			rng_ori = np.max(ori) - np.min(ori)
			if rng_ori == 0:
				rng_ori = 1

		C1 = (K1 * rng_ori) ** 2
		C2 = (K2 * rng_ori) ** 2

		mu_o = np.mean(ori)
		#var_o = np.var(ori)
		std_o = np.std(ori)
		A1 = 2 * mu_o ** 2 + C1
		A2 = 2 * std_o ** 2 + C2
		A = a * a * A1
		B = b * b * A2
		C = A / B 
		r1 = (np.sqrt((C + 1) ** 2 + 4 * C * (1.0 / ssim_target - 1)) - (C + 1) ) / (2 * C)
		r2 = C * r1 
		e_1 = a * ( np.sqrt( (mu_o * r1) ** 2 + A1 * r1) - mu_o * r1 * a )
		e_2 = b * ( np.sqrt( (std_o * r2) ** 2 + A2 * r2) - std_o * r2 * b )

		coeffs = [ssim_target, 2 * ssim_target * (mu_o * a + std_o * b), ssim_target * (A + B) + 4 * mu_o * std_o * a * b * (ssim_target - 1), 2 * a * b * (std_o * a * A1 + mu_o * b * A2) * (ssim_target - 1), A * B * (ssim_target - 1)]
		roots = np.roots(coeffs)
		eb = -np.inf 
		for root in roots:
			if not np.isnan(root) and not np.isinf(root) and not np.iscomplex(root) and root < 0 and root > eb:
				eb = np.real(root) 
		eb = -eb
		#print(eb)
		#print(e_1, e_2)
		
		return eb, min(e_1, e_2)

	def iter_blocks(ori: np.ndarray, block_size: int, step: int):
		ndim = ori.ndim
		shape = ori.shape
		ranges = [
			range(0, shape[d] - block_size + 1, step)
			for d in range(ndim)
		]
		for index in itertools.product(*ranges):
			slicer = tuple(
				slice(idx, idx + block_size)
				for idx in index
				)
			yield ori[slicer]




	if is_global:

		eb_1, eb_2 = calc_eb_by_ssim(ori_data, ssim_target, a, b)

		print("Global Stats:")
		print("Error bound 1: %.20g, Error bound 2: %.20g" % (eb_1, eb_2))
		

	else:
		ndim = ori_data.ndim
		shape = ori_data.shape
		ranges = [
			range(0, shape[d] - block_size + 1, block_size)
			for d in range(ndim)
		]
		num_blocks = prod(max(0, ((s - block_size) // shift + 1)) for s in shape)
		eb1s = np.zeros(num_blocks, dtype = np.double)
		eb2s = np.zeros(num_blocks, dtype = np.double)
		idx = 0
		rng = np.max(ori_data) - np.min(ori_data)
		for ori_block in iter_blocks(ori_data, block_size, shift):
			eb1, eb2 = calc_eb_by_ssim(ori_block, ssim_target, a, b, rng)
			eb1s[idx] = eb1
			eb2s[idx] = eb2
			idx += 1

		eb_rates = (eb2s - eb1s) / eb1s
		print("On blocks of side length %d and shift length %d:" % (block_size, shift))
		print("EB_accurate: max value: %.10g, min value: %.10g, mean value: %.10g, variance: %.10g, standard derivation: %.10g" % calc_stats_single(eb1s) )
		print("EB_approx: max value: %.10g, min value: %.10g, mean value: %.10g, variance: %.10g, standard derivation: %.10g" % calc_stats_single(eb2s) )
		print("EB_approx_error_rate: max value: %.10g, min value: %.10g, mean value: %.10g, variance: %.10g, standard derivation: %.10g" % calc_stats_single(eb_rates) )
		if write_to_file:
			eb1s.tofile(output_path+".eb_acc")
			eb2s.tofile(output_path+".eb_approx")
			print("%d ebs written to file." % num_blocks)
