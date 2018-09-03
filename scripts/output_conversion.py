import argparse
import json
import os


CLASS_NAMES = {0: 'BG', 1: 'pedestrian'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file')
    parser.add_argument('--converted_dir')
    args = parser.parse_args()

    try:
        output_dir = os.path.join(args.converted_dir)
        os.makedirs(output_dir)
    except OSError:
        print('Output folder already exists.')

    with open(args.output_file) as f:
        output = json.load(f)

    for item in output:
        """
            item[0]: abs fname
            item[1]: bb list
            item[2]: label list
            item[3]: score list
        """
        with open(os.path.join(args.converted_dir, os.path.basename(item[0]).split('.')[0] + '.txt'), 'w') as txt:
            for idx, pred in enumerate(item[1]):
                cls_name = CLASS_NAMES[item[2][idx]]
                txt.write(' '.join([cls_name, str(item[3][idx])] + list(map(str, item[1][idx]))) + '\n')

