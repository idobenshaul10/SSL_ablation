import numpy as np



class CIFAR100_mapper:
	def __init__(self):
		self.coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
								  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
								  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
								  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
								  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
								  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
								  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
								  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
								  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
								  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

		self.mapping = {c: np.where(self.coarse_labels == c)[0] for c in range(20)}

	def __call__(self, c):
		return self.mapping[c]



if __name__ == '__main__':
	a = CIFAR100_mapper()
	import pdb; pdb.set_trace()
