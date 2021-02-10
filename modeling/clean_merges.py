"""

Clean GPT-2 merges file, removing all tokens from the tokenizer that have
digits, other than the "0" - "9" tokens.

"""

merges_fname = "merges_gpt2.txt"
new_merges_fname = "merges_gpt2_single_digit_numbers.txt"

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

with open(new_merges_fname, 'w') as f_new:
    with open(merges_fname, 'r') as f:
        lines = f.read().split("\n")
        for l in lines:
            if len(l) < 1:
                break

            left, right = l.split(" ")
            if hasNumbers(left) or hasNumbers(right):
                print(left, right)
            else:
                f_new.write(l + "\n")
                

