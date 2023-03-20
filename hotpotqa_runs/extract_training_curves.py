import re
import sys

assert len(sys.argv) == 2
LOG_FILE = sys.argv[1]

def extract_digits(input_str):
    digit_regex = r'\d+'
    digit_strings = re.findall(digit_regex, input_str)
    digit_floats = [float(digit_str) for digit_str in digit_strings]
    return digit_floats

def main():
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    accuracies = []
    for line in lines:
        if "Trial summary:" in line:
            digits = extract_digits(line)
            accuracy = round(digits[0] / sum(digits), 2)
            accuracies += [accuracy]
    print(accuracies)

if __name__ == '__main__':
    main()
