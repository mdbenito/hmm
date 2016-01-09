import sys
if sys.version_info < (3, 5):
    print("Sorry, Python version >= 3.5 is required.")
    sys.exit(128)

import getopt
import inference
import data
from models.discrete import Multinomial, Poisson


def parse_options(argv):
    model = 'Multinomial'
    input_file = ''
    output_file = ''
    run_tests = False
    usage_str = "Usage: python hmm.py -i <input-file> -o <output-file>\n\n" \
                "Other options:\n" + \
                "    -h, --help           This help\n" + \
                "    -i, --input-file     Specifies the input file\n" + \
                "    -o, --output-file    Specifies the output file\n" + \
                "    -t, --run-tests      Run all tests\n" + \
                "    -m, --model          Model to use (one of Multinomial, Poisson)\n" + \
                "                         [Default: {0}]".format(model)
    try:
        opts, args = getopt.getopt(argv, 'hti:o:m:',
                                   ['help', 'run-tests', 'input-file=',
                                    'output-file=', 'model='])
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
        elif opt in ('-m', '--model'):
            model = str(arg)
    return [input_file, output_file, run_tests, model]


def main(argv):
    [input_file, output_file, run_tests, model] = parse_options(argv)
    if run_tests:
        import test_all
        test_all.main()
    else:
        d = data.load(input_file)
        if model.lower() is 'multinomial':
            m = Multinomial(d)
        elif model.lower() is 'poisson':
            m = Poisson(d)
        m = inference.iterate(m)
        inference.save(m, output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
