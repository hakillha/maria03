import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--clsfier_output')
	parser.add_argument('--gt_dir')
	args = parser.parse_args()