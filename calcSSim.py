import numpy as np 
import sys 
import itertools
from math import prod
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Usage: python calcSSIM.py [float/double] [ori_data_file] [decomp_data_file] [number_of_dims] [dims (slowest last)] [block_size (-1 for global)] [shift_size (optional, for blockwise)] [output_file (optional, for blockwise)]")
		print("Example 1: python calcSSIM.py float pressure.f32 pressure.f32.sz3.out 3 256 384 384 -1")
		print("Example 2: python calcSSIM.py float pressure.f32 pressure.f32.sz3.out 3 256 384 384 8 4 pressure_blockwise_ssim.dat")
		sys.exit(1)

	datatype = np.double if sys.argv[1] == "double" else np.float32 
	ori_data_path = sys.argv[2]
	decomp_data_path = sys.argv[3] 
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
	dec_data = np.fromfile(decomp_data_path, dtype = datatype).reshape(dims)

	def calc_stats_single(dat):
		mx = np.max(dat)
		mi = np.min(dat)
		mean = np.mean(dat)
		var = np.var(dat)
		std = np.sqrt(var)
		return mx, mi, mean, var, std

	def calc_ssim(ori, dec, rng_ori = None):
		K1 = 0.01
		K2 = 0.03
		if rng_ori == None:
			rng_ori = np.max(ori) - np.min(ori)
			if rng_ori == 0:
				rng_ori = 1

		C1 = (K1 * rng_ori) ** 2
		C2 = (K2 * rng_ori) ** 2
		mu_o = np.mean(ori)
		mu_d = np.mean(dec)
		var_o = np.var(ori)
		var_d = np.var(dec)

		cov_od = ((ori - mu_o) * (dec - mu_d)).sum() / ori.size

		luminance = (2 * mu_o * mu_d + C1) / (mu_o ** 2 + mu_d ** 2 + C1)

		contrast_x_structure = (2 * cov_od + C2) / (var_o + var_d + C2)
		
		return luminance * contrast_x_structure

	def iter_blocks(ori: np.ndarray, dec: np.ndarray, block_size: int, step: int):
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
			yield ori[slicer], dec[slicer]




	if is_global:

		ssim = calc_ssim(ori_data, dec_data)

		print("Global Stats:")
		print("Error bound: %.20g" % np.max(np.abs(ori_data - dec_data)))
		print("SSIM: %.10g" % ssim)

	else:
		ndim = ori_data.ndim
		shape = ori_data.shape
		ranges = [
			range(0, shape[d] - block_size + 1, block_size)
			for d in range(ndim)
		]
		num_blocks = prod(max(0, ((s - block_size) // shift + 1)) for s in shape)
		ssims = np.zeros(num_blocks, dtype = np.double)
		idx = 0
		#rng = np.max(ori_data) - np.min(ori_data)
		for ori_block, dec_block in iter_blocks(ori_data, dec_data, block_size, shift):
			cur_ssim = calc_ssim(ori_block, dec_block)
			ssims[idx] = cur_ssim
			idx += 1
		print("Global error bound: %.20g" % np.max(np.abs(ori_data-dec_data)))
		print("On blocks of side length %d and shift length %d:" % (block_size, shift))
		print("Blockwise SSIM: max value: %.10g, min value: %.10g, mean value (overall SSIM): %.10g, variance: %.10g, standard derivation: %.10g" % calc_stats_single(ssims) )

		if write_to_file:
			ssims.tofile(output_path)
			print("%d block SSIMs written to file." % num_blocks)
