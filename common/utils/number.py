import string

from word2number.w2n import word_to_num

month_dict = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 
    'may': 5, 'june': 6, 'july': 7, 'august': 8, 
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def get_number_from_word(word):
    if word.startswith('.'):
        try:
            number = float(word)
            return number
        except ValueError:
            return None
    puncts = string.punctuation.replace('-', '')
    word = word.strip(puncts)
    word = word.replace(',', '')
    if word == 'point':
        return None
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if word.lower() in month_dict:
                    number = month_dict[word.lower()]
                else:
                    number = None
    if number == float('nan') or number == float('inf'):
        return None
    return number