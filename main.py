import sys
if sys.version_info < (3, 5):
    print('Sorry, Python version >= 3.5 is required.')
    sys.exit(128)

import getopt
import inference
import data


def parse_options(argv):
    input_file = ''
    output_file = ''
    usage_str = 'hmm.py -i <input_file> -o <output_file>'
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["input_file=", "output_file="])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage_str)
            sys.exit()
        elif opt in ("-i", "--input_file"):
            input_file = arg
        elif opt in ("-o", "--output_file"):
            output_file = arg
    return [input_file, output_file]
    # print('Input file is "', input_file)
    # print('Output file is "', output_file)


def main(argv):
    [input_file, output_file] = parse_options(argv)
    # d = data.load_data(input_file)
    d = data.generate()
    m = inference.iterate(d, maxiter=5)

    # inference.viterbi_path()

if __name__ == '__main__':
    main(sys.argv[1:])
