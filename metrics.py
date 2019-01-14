from __future__ import division
import sys, re, string, nltk, codecs
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
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
    global CONTRACTION_DICTIONARY
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
    # print(initialTokens) # TEST
    processedTokens = removePunctuationTokens(initialTokens)
    return processedTokens


# File -> String
# Takes in a file, delete double quotes and convert utf-8 single quotes to ', return revised content
# Ex: '\xe2\x80\x9cI said \xe2\x80\x98books,\xe2\x80\x99 \xe2\x80\x9d she said.\n' -> '"I said 'books,' " she said.\n'
def filterSpecialPunctuation(txtFile):
    contents = txtFile.read()
    # Turn unicode single quotes to ascii single quotes (for contraction processing purposes)
    contents = re.sub(u"(\u2019|\u2018)", "'", contents)
    # Get rid of double quotes, em dashes, double hyphens, and underscores
    contents = re.sub(u"(\u201c|\u201d|\u2014|\u005f|\u2010\u2010)", ' ', contents)
    return contents


# File -> String
# Function opens and reads a .txt file, delete double quotes from the contents, and returns the contents of file as a string
def getTxtFromFile():
    # txtFileName = raw_input("Text file name (.txt): ")
    txtFile = codecs.open('sample.txt', encoding='utf-8')
    contents = filterSpecialPunctuation(txtFile)
    return contents


# (String, String) -> String
# Determine a flag for the token based on it's tag (ADJ, NOUN, VERB, ADV, else) so that lemmatizer can lemmatize properly
# Ex: ('purple', 'ADJ') -> 'a'
def assignPosTag(tuple):
    posTag = 'n' # Default: treat as noun
    if tuple[1] == 'ADJ':
        posTag = 'a'
    elif tuple[1] == 'ADV':
        posTag = 'r'
    elif tuple[1] == 'VERB':
        posTag = 'v'
    return posTag


# List String -> Int
# Counts the number of unique lemmatized words in the list of tokens (only up to the 55,000th token)
def countLemmatizedWords(tokens):
    lemmatizedWordDict = {}
    PosTagList = nltk.pos_tag(tokens, tagset='universal')
    # print(PosTagList) # TEST
    for index, tokenAndPosTuple in enumerate(PosTagList):
        if not index < 55000:
            break
        posTag = assignPosTag(tokenAndPosTuple)
        lemmatizedToken = lemmatizer.lemmatize(tokenAndPosTuple[0], pos=posTag)
        # print(lemmatizedToken) # TEST
        if lemmatizedToken not in lemmatizedWordDict:
            lemmatizedWordDict[lemmatizedToken] = True
    return len(lemmatizedWordDict)
 

# List String -> Int
# Divides number of lemmatized word types (total number of different words) by total word tokens (up to 55,000), and returns this ratio
def getTypeTokenRatio(tokens):
    numWordTokens = len(tokens)
    # print("Length of tokens:", numWordTokens) # TEST
    if numWordTokens > 55000:
        numWordTokens = 55000
    numWordTypes = countLemmatizedWords(tokens)
    # print("Total word types:", numWordTypes) # TEST
    # print("Modified length of tokens:", numWordTokens) # TEST
    TTRRatio = (numWordTypes / numWordTokens)
    # print("TTR Ratio:", TTRRatio) # TEST
    return TTRRatio


# (String, String) -> Bool
# Determine if the word is a content word by matching it's ID to either a noun, adj, adverb, or verb
# Ex: ('purple', 'ADJ') -> True, ('the', 'DET') -> False
def isContentWord(tuple):
    if re.match('ADJ|ADV|NOUN|VERB', tuple[1]):
        return True
    return False


# List (String, String) -> 
# Lemmatize each token in the list of tuples (token, posTag), modify list tuples in place
def makeLemmatizedList(listTuples):
    lemmatizedList = []
    for t in listTuples:
        posTag = assignPosTag(t)
        t = (lemmatizer.lemmatize(t[0], pos=posTag), t[1])
        lemmatizedList.append(t)
    return lemmatizedList


# List (String, String) -> List String
# Add tokens to a list if that token is a noun, adjective, verb, or adverb
# Ex: [('the', 'DET'), ('purple', 'ADJ'), ('clown', 'NOUN')] -> ['purple', 'clown']
def removeNonContentWords(listTags):
    contentWordsList = []
    for tuple in listTags:
        if isContentWord(tuple):
            contentWordsList.append(tuple[0]) 
            # print("Appended", tuple) # TEST
    return contentWordsList


# List String -> List String
# Make a list of tokens where each token must be a lemmatized content word (noun, adjective, verb, or adverb)
# Ex: ["Here", "is", "a", "shiny", "balloons", "of", "death"] -> ["Here", "is", "shiny", "balloon", "death"]
def getLemmatizedContentTokens(tokens):
    newList = nltk.pos_tag(tokens, tagset='universal') # Tags: [('purple', 'ADJ'), ('the', 'DET')]
    # print("Tagged tokens list:", newList) # TEST
    newList = makeLemmatizedList(newList)
    # print("Lemmatized list:", newList) # TEST
    newList = removeNonContentWords(newList)
    # print("Only content words:", newList) # TEST
    return newList


# List String -> Int
# Count the number of repetitions for each token within 10 subsequent tokens, and add it all up.
# Ex: ["got", "door", "stopped", "suddenly", "then", "walked", "Look", "something", "bundle", "clothes", "lying", "door", "something", "pulled", "Mathilde", "thought", "look", "Tuppence", "wondered", "quickened", "pace", "almost", "running", "got", "door", "stopped", "suddenly", "was", "bundle", "old", "clothes", "clothes", "was", "old", "enough", "so", "was", "body", "wore", "Tuppence", "bent", "over", "then", "stood", "steadied", "hand", "door"] -> 7
def countRepetitions(tokens):
    dictRepetitions = {}
    hashTable = {}
    totalValue = 0
    for index, token in enumerate(tokens):
        token = token.lower() # Dict is case-sensitive, make sure tokens are uniform
        if hashTable.get(token) != None:
            hashTable[token] += 1
            if dictRepetitions.get(token) != None:
                dictRepetitions[token] += 1
            else:
                dictRepetitions[token] = 1
        else:
            hashTable[token] = 1
        if index >= 10:
            tokenToRemove = tokens[index - 10].lower()
            hashTable[tokenToRemove] -= 1
            if hashTable[tokenToRemove] <= 0: 
                del hashTable[tokenToRemove]
    for value in dictRepetitions.itervalues():
        totalValue += value
    return totalValue


# List String -> Int
# Returns the percentage of close distance repetitions (within 10 lemmatized content words) in the first first 55,000 tokens of the text
# By computing close distance repetitions / number of content words in the first 55000 tokens
def getWordRepetitionPercent(tokens):
    if len(tokens) > 55000:
        tokens = tokens[:55000]
    lemmatizedContentTokens = getLemmatizedContentTokens(tokens)
    # print(lemmatizedContentTokens) # TEST
    numRepetitions = countRepetitions(lemmatizedContentTokens)
    # print(numRepetitions) # TEST
    # print(len(lemmatizedContentTokens)) # TEST
    repetitionPercent = numRepetitions / len(lemmatizedContentTokens)
    return repetitionPercent


def main():
    txt = getTxtFromFile()
    tokens = tokenizeTxt(txt)
    # print(tokens) # TEST

    typeTokenRatio = getTypeTokenRatio(tokens)
    print("TTR:", typeTokenRatio)

    wordRepetitions = getWordRepetitionPercent(tokens)
    print("Repetition Percent:", wordRepetitions)

    return


if __name__ == "__main__":
    main()