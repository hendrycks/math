def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

class NotEqual:
    def __eq__(self, other):
        return False

def strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = remove_right_units(string)


    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

if __name__ == "__main__":
    """
    test_in = "\\tfrac{1}{2} + \\frac1{72}"
    test_out = "\\\\frac{1}{2} + 2/3"
    print(is_equiv(test_in, test_out), "Expected", False)

    test_in = "\\tfrac{1}{2} +\\! \\frac1{72}"
    test_out = "\\\\dfrac{1}{2} +\\frac{1}{72}"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "10\\text{ units}"
    test_out = "10 "
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "10\\text{ units}"
    test_out = "100 "
    print(is_equiv(test_in, test_out), "Expected", False)

    test_in = "10"
    test_out = "\\$10"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "\\left(x-2\\right)\\left(x+2\\right)"
    test_out = "(x-2)(x+2)"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "0.1"
    test_out = ".1"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "10\\%"
    test_out = "10"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "10\\sqrt{3} + \\sqrt4"
    test_out = "10\\sqrt3 + \\sqrt{4}"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "\\frac34i"
    test_out = "\\frac{3}{4}i"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "\\tfrac83"
    test_out = "\\frac{8}{3}"
    print(is_equiv(test_in, test_out), "Expected", True)

    test_in = "5x - 7y + 11z + 4 = 0"
    test_out = "x + y - z + 2 = 0"
    print(is_equiv(test_in, test_out), "Expected", False)
    
    test_in = "1/2"
    test_out = "\\frac{1}{2}"
    print(is_equiv(test_in, test_out), "Expected", True)
    """


