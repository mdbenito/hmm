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
    run_tests = False
    usage_str = 'Usage: python hmm.py -i <input-file> -o <output-file>\n\nOther options:\n' + \
                '\t-h, --help\t\tThis help\n' + \
                '\t-i, --input-file\tSpecifies the input file\n' + \
                '\t-o, --output-file\tSpecifies the output file\n' + \
                '\t-t, --run-tests\t\tRun all tests\n'
    try:
        opts, args = getopt.getopt(argv, 'hti:o:',
                                   ['help', 'run-tests', 'input-file=', 'output-file='])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt in ('-i', '--input-file'):
            input_file = arg
        elif opt in ('-o', '--output-file'):
            output_file = arg
        elif opt in ('-t', '--run-tests'):
            run_tests = True
    return [input_file, output_file, run_tests]
    # print('Input file is "', input_file)
    # print('Output file is "', output_file)


def main(argv):
    [input_file, output_file, run_tests] = parse_options(argv)
    if run_tests:
        import test_all
        test_all.main()
    else:
        d = data.load(input_file)
        d = data.generate()
        m = inference.iterate(d)
        inference.save(m, output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
