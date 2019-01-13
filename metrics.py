import sys, re, string, nltk, codecs

reload(sys)
sys.setdefaultencoding("utf-8") # Instead of ASCII, which had caused errors with nltk.tokenizer as it processed file contents

# Dictionary of contractions
CONTRACTION_DICTIONARY = {
    "'t": True,
    "'ve": True,
    "'s": True,
    "'d": True,
    "'ll": True,
    "'re": True,
    "'m": True,
    "'clock": True,
    "'er": True,
    "'twas": True,
    "'tis": True
}


# String -> String
# Remove single quote from token IF it is NOT the end of a contraction (e.g. 't from don't, 've from I've, etc)
# Ex: "'book" -> "book", "'ve" -> "'ve"
def deleteSingleQuoteIfNotContraction(token):
    if CONTRACTION_DICTIONARY.get(token) == None:
        token = re.sub(r"'", '', token)
    return token


# List String -> List String
# Remove all punctuation tokens from the list, and return the processed list.
# Ex: ["Hello", "!", "I", "'m", "happy", "with", "the", "blue-ish", "pony", ",", maybe", "?"] -> ["Hello", "I", "'m", "happy", "with", "the", "blue-ish", "pony", "maybe"] 
def removePunctuationTokens(tokens):
    processedTokens = []
    for t in tokens:
        if not re.match("[" + string.punctuation + "]+", t): # Add to processedTokens anything that is not punctuation
            processedTokens.append(t)
        elif re.match('\'[A-Za-z]+', t): #Ex: Checks special case for single quotes: matches 'book, or 've, but not ' 
            t = deleteSingleQuoteIfNotContraction(t)
            processedTokens.append(t)
    return processedTokens


# String -> List String
# Tokenize text (separate text into words, removing punctuation and separating contractions). Return resulting tokens.
# Ex: "Hello! I'm happy with the blue-ish pony, maybe?" -> ["Hello", "I", "'m", "happy", "with", "the", "blue-ish", "pony", "maybe"]
def tokenizeTxt(text):
    initialTokens = nltk.word_tokenize(text) 
    print(initialTokens) # TEST
    processedTokens = removePunctuationTokens(initialTokens)
    return processedTokens


# File -> String
# Takes in a file, delete double quotes and convert utf-8 single quotes to ', return revised content
# Ex: '\xe2\x80\x9cI said \xe2\x80\x98books,\xe2\x80\x99 \xe2\x80\x9d she said.\n' -> '"I said 'books,' " she said.\n'
def filterSpecialPunctuation(txtFile):
    contents = txtFile.read()
    # Turn unicode single quotes to ascii single quotes (for contraction processing purposes)
    contents = re.sub(u"(\u2019|\u2018)", "'", contents)
    # Get rid of double quotes, em dashes, and underscores
    contents = re.sub(u"(\u201c|\u201d|\u2014|\u005f)", ' ', contents)
    return contents


# File -> String
# Function opens and reads a .txt file, delete double quotes from the contents, and returns the contents of file as a string
def getTxtFromFile():
    # txtFileName = raw_input("Text file name (.txt): ")
    txtFile = codecs.open('sample.txt', encoding='utf-8')
    contents = filterSpecialPunctuation(txtFile)
    return contents


def main():
    txt = getTxtFromFile()
    tokens = tokenizeTxt(txt)
    print(tokens)
    return


if __name__ == "__main__":
    main()