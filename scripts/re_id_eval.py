import argparse
import json

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--query_file')
	parser.add_argument('--gallery_file')
	args = parser.parse_args()

	with open(args.gallery_file, 'r') as gallery_file:
		gallery_list = json.load(gallery_file)
		
		for frame in gallery_list:
			print(len(frame[4]))
			
	with open(args.query_file, 'r') as query_file:
		query_list = json.load(query_file)

		for query in query_list:
			print(len())
		