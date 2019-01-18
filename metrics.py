from __future__ import division, print_function
import sys, os, re, string, nltk, codecs, random
import numpy as np
from nltk.stem import WordNetLemmatizer
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.layouts import column, row, widgetbox
from bokeh.models import HoverTool
from bokeh.models.widgets import Paragraph, Panel, Tabs
from bokeh.palettes import Viridis, Plasma 


lemmatizer = WordNetLemmatizer()


reload(sys)
sys.setdefaultencoding("utf-8") # Instead of ASCII, which had caused errors with nltk.tokenizer as it processed file contents


# TextData stores the metrics for a given text:
class TextData:

    def __init__(self, title, year, ttrRatio, listWTIR, repetitionPercent):
        self.title = title
        self.year = year
        self.ttrRatio = ttrRatio # Type to token ratio
        self.listWTIR = listWTIR
        self.repetitionPercent = repetitionPercent

    def printTextData(self):
        print(self.title + ":", str(self.year) + ",", str(self.ttrRatio) + ",", str(self.listWTIR) + ",", str(self.repetitionPercent)) 


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









### GENERAL FUNCTIONS: ###

# (String, String) -> String
# Determine a flag for the token based its part of speech (ADJ, NOUN, VERB, ADV, else) so that lemmatizer.lemmatize(...) can lemmatize properly
# Ex: ('purple', 'ADJ') -> 'a'. Can then call lemmatizer.lemmatize('purple', pos='a')
def assignPosTag(tuple):
    posTag = 'n' # Default: treat as a noun
    if tuple[1] == 'ADJ':
        posTag = 'a'
    elif tuple[1] == 'ADV':
        posTag = 'r'
    elif tuple[1] == 'VERB':
        posTag = 'v'
    return posTag


# List TextData -> 
# Prints the content of each TextData in the list
def printListTextData(listTextData):
    for td in listTextData:
        td.printTextData()
    return 









### FUNCTIONS FOR TOKEN/TYPE RATIO: ###

# String -> String
# Token processing function: Remove single quote from token (str) IF it is NOT the end of a contraction (e.g. will not remove 't from don't, 've from I've, etc)
# Ex: "'book" -> "book", "'ve" -> "'ve"
def deleteSingleQuoteIfNotContraction(token):
    global CONTRACTION_DICTIONARY
    if CONTRACTION_DICTIONARY.get(token) == None:
        token = re.sub(r"'", '', token)
    return token


# List String -> List String
# Token processing function: Remove all punctuation tokens from the list, and return the processed list.
# Ex: ["Hello", "!", "I", "'m", "happy", "with", "the", "blue-ish", "pony", ",", maybe", "?"] -> ["Hello", "I", "'m", "happy", "with", "the", "blue-ish", "pony", "maybe"] 
def removePunctuationTokens(tokens):
    processedTokens = []
    for t in tokens:
        if not re.match("[" + string.punctuation + "]+", t): # Add to processedTokens anything that is not punctuation
            processedTokens.append(t)

        # Checks special case for single quotes: matches 'book, or 've, but not '
        elif re.match('\'[A-Za-z]+', t):  
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
    # print(processedTokens) # TEST
    return processedTokens


# File -> String
# Text processing: Takes in a file, delete double quotes and convert utf-8 single quotes to ', return revised content
# Ex: '\u201cI said \u2018books,\u2019 \u201d she said.' -> '"I said 'books,' " she said.'
def filterSpecialPunctuation(txtFile):
    contents = txtFile.read()

    # Turn unicode single quotes to ascii single quotes (for contraction processing purposes)
    contents = re.sub(u"(\u2019|\u2018)", "'", contents)

    # Get rid of double quotes, em dashes, double hyphens, underscores, and ellipses
    contents = re.sub(u"(\u201c|\u201d|\u2014|\u005f|\u2010\u2010|\u2026)", ' ', contents)
    return contents


# File -> String
# Process text by dealing with special cases of punctuation, and return the processed text. 
def getTxtFromFile(txtFile):
    contents = filterSpecialPunctuation(txtFile)
    return contents


# List String -> Int, List (Int, Int)
# Counts the total number of unique lemmatized words in the list of tokens (only up to the 55,000th token). 
# Also calculates the WTIR (TOTAL unique lemmatized word tokens calculated at every 10,000th interval)
def countLemmatizedWordsAndWTIR(tokens):
    lemmatizedWordDict = {}

    # Measures unqiue lemmatized words up to the 55,000th token
    totalWordTypes = 0 
    reached55000Token = False

    # Word type introduction rate list, where each item is a tuple (num unique word types, interval) measuring the total unique word types encountered at every 10,000th interval
    listWTIR = [(0, 0)]

    # Creates list of tuples out of the list of tokens: [(token, pos tag), etc]
    PosTagList = nltk.pos_tag(tokens, tagset='universal') 
    # print(PosTagList) # TEST

    for index, tokenPosTuple in enumerate(PosTagList):
        if not index < 55000: # Only consider up to the 55,000th token
            totalWordTypes = len(lemmatizedWordDict)
            reached55000Token = True
        posTag = assignPosTag(tokenPosTuple)
        lemmatizedToken = lemmatizer.lemmatize(tokenPosTuple[0], pos=posTag)
        # print(lemmatizedToken) # TEST
        if lemmatizedToken not in lemmatizedWordDict:
            lemmatizedWordDict[lemmatizedToken] = True
        # Determine if must calculate the WTIR at an interval of 10,000
        if (index + 1) % 10000 == 0: 
            newWTIR = (index + 1, len(lemmatizedWordDict)) # Ex: (10000, 2000)
            listWTIR.append(newWTIR)
    
    # Deals with cases where the num of tokens is less than 55,000
    if not reached55000Token:
        totalWordTypes = len(lemmatizedWordDict)
    return totalWordTypes, listWTIR
 

# List String -> Int, List (Int, Int)
# Divides number of lemmatized word types (total number of different words) by total word tokens (up to 55,000), and returns this ratio.
# Also returns a list of the word type introduction rates (TOTAL unique lemmatized word tokens calculated at every 10,000th interval)
# This is effectively a measure of vocabulary size in a given text
def getTTRAndWTIR(tokens):
    numWordTokens = len(tokens)
    # print("Length of tokens:", numWordTokens) # TEST
    if numWordTokens > 55000:
        numWordTokens = 55000
    numWordTypes, listWTIR = countLemmatizedWordsAndWTIR(tokens)
    # print("Total word types:", numWordTypes) # TEST
    # print("Modified length of tokens:", numWordTokens) # TEST
    TTRRatio = (numWordTypes / numWordTokens)
    # print("TTR Ratio:", TTRRatio) # TEST
    return TTRRatio, listWTIR










### FUNCTIONS FOR WORD REPETITION METRIC: ###

# (String, String) -> Bool
# Determine if the word is a content word by matching it's tag (part of speech tag) to either a noun, adj, adverb, or verb
# Ex: ('purple', 'ADJ') -> True, ('the', 'DET') -> False
def isContentWord(tuple):
    if re.match('ADJ|ADV|NOUN|VERB', tuple[1]):
        return True
    return False


# List (String, String) -> List (String, String)
# Lemmatize each token in the list of tuples (token, posTag), return lemmatized token/tag tuple
def makeLemmatizedTupleList(listTuples):
    lemmatizedList = []
    for t in listTuples:
        posTag = assignPosTag(t) # Assign a pos flag based on the type of token it is (noun, verb, etc)
        t = (lemmatizer.lemmatize(t[0], pos=posTag), t[1]) # Tuple with lemmatized token and tag
        lemmatizedList.append(t)
    return lemmatizedList


# List (String, String) -> List String
# Create a list of tuples that filters out the tuples (token, tag) whose token is not a content word
# Ex: [('the', 'DET'), ('purple', 'ADJ'), ('clown', 'NOUN')] -> ['purple', 'clown']
def removeNonContentWords(listTags):
    contentWordsList = []
    for tuple in listTags:
        if isContentWord(tuple):
            contentWordsList.append(tuple[0]) 
            # print("Appended", tuple) # TEST
    return contentWordsList


# List String -> List String
# Make a list of tokens where each token must be a lemmatized content word (noun, adjective, verb, or adverb). 
# Process: add tags to each token, lemmatize the token in each tuple, remove tuples with non-content tokens, turn list tuples back into list strings (tokens)
# Ex: ["Here", "is", "a", "shiny", "balloons", "of", "death"] -> ["Here", "is", "shiny", "balloon", "death"]
def getLemmatizedContentTokens(tokens):
    # Add tags to each token: List String -> List (String, String)
    newList = nltk.pos_tag(tokens, tagset='universal') # Tag: [('purple', 'ADJ'), ('the', 'DET')]
    # print("Tagged tokens list:", newList) # TEST

    # Lemmatize the token in each tuple
    newList = makeLemmatizedTupleList(newList)
    # print("Lemmatized list:", newList) # TEST

    # Remove tuples with non-content tokens, and condense list tuples into list tokens
    newList = removeNonContentWords(newList)
    # print("Only content words:", newList) # TEST
    return newList


# List String -> Int
# Count the number of repetitions for each token within 10 subsequent tokens, and add it all up.
# Ex: ["got", "door", "stopped", "suddenly", "then", "walked", "Look", "something", "bundle", "clothes", "lying", "door", "something", "pulled", "Mathilde", "thought", "look", "Tuppence", "wondered", "quickened", "pace", "almost", "running", "got", "door", "stopped", "suddenly", "was", "bundle", "old", "clothes", "clothes", "was", "old", "enough", "so", "was", "body", "wore", "Tuppence", "bent", "over", "then", "stood", "steadied", "hand", "door"] -> 7
def countRepetitions(tokens): 
    hashTable = {}
    totalRepetitions = 0
    
    # Identify repetitions
    for index, token in enumerate(tokens):
        token = token.lower() # Dict is case-sensitive, make sure tokens are uniform
        if hashTable.get(token) != None: #If token is already in hash table, make sure the program knows it has appeared multiple times
            hashTable[token] += 1
            totalRepetitions += 1
        else:
            hashTable[token] = 1
        if index >= 10: # Only looking for repetitions within 10 subsequent tokens
            tokenToRemove = tokens[index - 10].lower()
            hashTable[tokenToRemove] -= 1
            if hashTable[tokenToRemove] <= 0: 
                del hashTable[tokenToRemove]
    return totalRepetitions


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

    repetitionPercent = (numRepetitions / len(lemmatizedContentTokens)) * 100
    return repetitionPercent









### TEXT PROCESSING FUNCTIONS: ###

# List TextData -> 
# Write metrics generated by program to a text file (input by user). If file does not exist, program will create a file
# located in the directory of this program
def writeTextDataToFile(listTextData):
    fileName = raw_input("File to export data to (.txt): ")
    txtFile = open(fileName, "w")
    for td in listTextData:
        txtFile.write(td.title + ' ' + str(td.year) + ' ' + str(td.ttrRatio) + ' ' + str(td.listWTIR) + ' ' + str(td.repetitionPercent))
    txtFile.close()
    print("Data written to", fileName)
    return 


# String String -> File
# Text processing: Function opens and reads a .txt file, delete double quotes from the contents, and returns the contents of file as a string
def openFile(txtFileName, directoryName):
    currentPath = os.getcwd()
    fullPath = os.path.join(currentPath, directoryName, txtFileName)
    txtFile = codecs.open(fullPath, encoding='utf-8')
    return txtFile


# String -> String, Int
# Parses the text file name and returns the year the text was written as an int, and return the title of the text as a string
# Ex: "1920_TheMysteriousAffairAtStyles_AC.txt" -> "The Mysterious Affair At Styles", 1920
def parseFileNameForDateAndTitle(fileName):
    listComponents = fileName.split("_")
    year = int(listComponents[0])
    title = " ".join(re.sub(r"([A-Z])", r" \1", listComponents[1]).split())
    return title, year


# -> String
# Return the name of the directory the program is running in
def getDirectoryName():
    directoryName = raw_input("Enter folder name where text samples are located: ")
    return directoryName


# String -> List String
# Pull out all the text files names in the directory that stores sample writing, return as a list of strings
def getTxtFileNames(directoryName):
    listFileNames = []
    for txtFile in os.listdir(directoryName):
        listFileNames.append(txtFile)
    # print(listFileNames) # TEST
    return listFileNames


# String, String -> List Int, List (Int, Int), List Int
# Analyze a text file for the 3 metrics
def analyzeTextFile(fileName, directoryName):
    file = openFile(fileName, directoryName)
    text = getTxtFromFile(file)
    tokens = tokenizeTxt(text)
    typeTokenRatio, listWTIR = getTTRAndWTIR(tokens)
    wordRepetitions = getWordRepetitionPercent(tokens)
    return typeTokenRatio, listWTIR, wordRepetitions


# None -> List TextData
# Get TextData for each text file in a user-specified directory
def getTextData():
    directoryName = getDirectoryName()
    txtFileNames = getTxtFileNames(directoryName)
    listTextData = []
    for fileName in txtFileNames:
        typeTokenRatio, listWordTypeIntroductionRate, repetitionPercent = analyzeTextFile(fileName, directoryName)
        title, year = parseFileNameForDateAndTitle(fileName)
        td = TextData(title, year, typeTokenRatio, listWordTypeIntroductionRate, repetitionPercent)
        listTextData.append(td)
    return listTextData









### GRAPHING FUNCTIONS: ###

# List Int, List Int -> (Int, Int)
# Estimate coefficients for linear regression model, where y = b0 + b1*x
# Code from https://www.geeksforgeeks.org/linear-regression-python-implementation/
def getCoefficients(xList, yList):
    numPoints = np.size(xList) # Number of points in data set 
    xMean, yMean = np.mean(xList), np.mean(yList)
    crossDeviationXY = np.sum(yList * xList) - (numPoints * xMean * yMean)
    crossDeviationXX = np.sum(xList * xList) - (numPoints * xMean * xMean)
    b1 = crossDeviationXY / crossDeviationXX
    b0 = yMean - (b1 * xMean)
    return (b0, b1)


# List TextData -> List Int
# Extract year of each TextData, create a list of years, return that list
def getListYears(listTextData):
    listYear = []
    for td in listTextData:
        listYear.append(td.year)
    return listYear


# List TextData -> List Float
# Extract TTR of each TextData, create a list of TTR, return that list
def getListTTR(listTextData):
    listTTR = []
    for td in listTextData:
        listTTR.append(td.ttrRatio)
    return listTTR


# List TextData -> List Float
# Extract repetition percent of each TextData, create a list of repetition percent, return that list
def getListRepetitionPercent(listTextData):
    listRepetitionPercent = []
    for td in listTextData:
        listRepetitionPercent.append(td.repetitionPercent)
    return listRepetitionPercent


# List TextData -> List Float
# Extract title of each TextData, create a list of titles, return that list
def getListTitles(listTextData):
    listTitle = []
    for td in listTextData:
        listTitle.append(td.title)
    return listTitle


# String -> String
# Generates a random color in either the Plasma or Viridis palette.
# colorGenerator("Viridis") -> '#1F958B'
def colorGenerator(paletteName):
    indexOneList = [256, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    firstIndex = random.choice(indexOneList)
    secondIndex = random.randint(0, firstIndex - 1)
    if paletteName == "Viridis":
        return Viridis[firstIndex][secondIndex]
    return Plasma[firstIndex][secondIndex]


# Int, Int -> String
# Generate a string form of a linear equation given its slope and intercerpt
def getEquation(b0, b1):
    string = "y = " + str(b1) + "x + " + str(b0)
    return string


# List Int, List Int, List String -> Plot
# Create a plot containing information for the type-to-token ratio (line of best fit, scatterplot)
def createTTRGraph(xPosnList, yPosnList, listTitle):
    xArr, yArr = np.array(xPosnList), np.array(yPosnList)
    
    source = ColumnDataSource(data=dict(
    x = xPosnList,
    y = yPosnList,
    title = listTitle))

    tooltips = [
    ("Year Published", "@x"),
    ("TTR Ratio", "@y"),
    ("Text Title", "@title")
    ]

    graph = figure(title = "Type-to-Token Ratio: Measure of Vocabulary Size", 
            x_axis_type = "linear",
            y_axis_type = "linear",
            x_range = (xArr[0] - 5, xArr[-1] + 5),
            y_range = (0, .15),
            x_axis_label = "Year Published",
            y_axis_label = "Type-to-Token Ratio",
            tooltips = tooltips)
    
    # Equation coefficients for line of best fit:
    b0 = getCoefficients(xArr, yArr)[0]
    b1 = getCoefficients(xArr, yArr)[1]

    # Graph line of best fit:
    graph.line(xArr, b0 + (b1 * xArr), legend = getEquation(b0, b1), line_color = colorGenerator("Viridis"), line_dash = "solid", line_width = 3.0)

    # Graph individual data points:
    graph.circle(x = 'x', y = 'y', size = 15, source = source, fill_color = colorGenerator("Plasma"), line_color = colorGenerator("Plasma"))

    graph.legend.location = "top_right"
    return graph


# List Int, List Int, List String -> Plot
# Create a Plot for word repetition percent. 
def createWordRepetitionGraph(xPosnList, yPosnList, listTitle):
    xArr, yArr = np.array(xPosnList), np.array(yPosnList)
    
    source = ColumnDataSource(data=dict(
    x = xPosnList,
    y = yPosnList,
    title = listTitle))

    tooltips = [
    ("Year Published", "@x"),
    ("Repetition Percent", "@y"),
    ("Text Title", "@title")
    ]

    graph = figure(title = "Close Distance Word Repetition Percent", 
            x_axis_type = "linear",
            y_axis_type = "linear",
            x_range = (xArr[0] - 5, xArr[-1] + 5),
            y_range = (0, 20),
            x_axis_label = "Year Published",
            y_axis_label = "Lexical Repetition as Percentage",
            tooltips = tooltips)
    
    # Equation coefficients for line of best fit:
    b0 = getCoefficients(xArr, yArr)[0]
    b1 = getCoefficients(xArr, yArr)[1]

    # Graph line of best fit:
    graph.line(xArr, b0 + (b1 * xArr), legend = getEquation(b0, b1), line_color = colorGenerator("Viridis"), line_dash = "dashed", line_width = 3.0)

    # Graph individual data points:
    graph.circle(x = 'x', y = 'y', size = 15, source = source, fill_color = colorGenerator("Plasma"), line_color = colorGenerator("Plasma"))

    graph.legend.location = "top_right"
    return graph


# TextData -> List Int, List Int
# Extract listWTIR from td, and return as 2 lists. 
# List 1 and 2 contain the first and second values (respectivelyof the tuples in listWTIR.
def getListWTIR(textData):
    listInterval = []
    listToken = []
    for wtir in textData.listWTIR:
        listInterval.append(wtir[0])
        listToken.append(wtir[1])
    return listInterval, listToken


# Int, List Int -> String, String
# Assigns a legend name and a line_dash name for a data point based off the year the text was written
def assignLegendCategoryAndLineDash(year, listYear):
    interval = (listYear[-1] - listYear[0]) / 3
    if year <= listYear[0] + interval:
        return "Early Works", "solid"
    elif year <= listYear[0] + (2 * interval):
        return "Mid-career Works", "dotted"
    return "Later Work", "dotdash"


# List TextData -> Int, Int
# Find the maximum token interval and maximum word type token count in each TextData.listWTIR of the list TextData
def maxTokenIntervalAndWordType(listTextData):
    maxTokenInterval = 0
    maxWordType = 0
    for td in listTextData:
        currentTokenInterval, currentWordType = td.listWTIR[-1][0], td.listWTIR[-1][1]
        if currentTokenInterval > maxTokenInterval:
            maxTokenInterval = currentTokenInterval
        if currentWordType > maxWordType:
            maxWordType = currentWordType
    return maxTokenInterval, maxWordType


# List Text Data, List Int, List String, String -> Panel 
# Create a Panel for the WTIR. Either a line graph or circle graph specified by mode ("line" or "circle")
def createWTIRGraph(listTextData, listYear, listTitle, mode):
    # Value Int -> List Value
    # Create a list of values that is repeated n times
    # Ex: createListRepeats(5, 5) -> [5, 5, 5, 5, 5]
    def createListRepeats(value, n):
        list = []
        while n > 0:
            list.append(value)
            n = n - 1
        return list

    earlyCategoryColor, midCategoryColor, lateCategoryColor = colorGenerator("Plasma"), colorGenerator("Plasma"), colorGenerator("Plasma")

    # String -> String
    # Assign a color for the circle graph mode based on what category the data point belongs to
    def assignCircleColor(legendCategoryName):
        if legendCategoryName == "Early Works":
            return earlyCategoryColor
        elif legendCategoryName == "Mid-career Works":
            return midCategoryColor
        return lateCategoryColor

    maxTokenInterval, maxWordType = maxTokenIntervalAndWordType(listTextData)
    graph = figure(title = "Word Types Introduced per 10,000 Tokens",
                x_axis_type = "linear",
                y_axis_type = "linear",
                x_range = (0, maxTokenInterval + 5000),
                y_range = (0, maxWordType + 1000),
                x_axis_label = "Token Interval",
                y_axis_label = "Word Types Introduced")
                
    for td in listTextData:
        xPosnList, yPosnList = getListWTIR(td)
        
        source = ColumnDataSource(data = dict(
            x = xPosnList,
            y = yPosnList,
            year = createListRepeats(td.year, len(xPosnList)), # Must do this because ColumnDataSource's columns must be of the same length
            title = createListRepeats(td.title, len(xPosnList)) 
        ))

        graph.add_tools(HoverTool(
            tooltips = [
                ("Year Published", "@year"),
                ("Text Title", "@title"),
                ("Total Word Types", "@y"),
                ("Token Interval", "@x")
            ],
            formatters = {
                "Text Title" : "printf"
            },
            mode = "mouse"
        ))

        legendCategoryName, lineDashName = assignLegendCategoryAndLineDash(td.year, listYear)
        if mode == "line":
            color = colorGenerator("Viridis")
            graph.line(x = 'x', y = 'y', line_width = 2, color = color, muted_color = color, muted_alpha = .1, legend = legendCategoryName, line_dash = lineDashName, source = source)
        else:
            graph.circle(x = 'x', y = 'y', size = 10, source = source, fill_color = assignCircleColor(legendCategoryName), muted_color = assignCircleColor(legendCategoryName), muted_alpha = .1, legend = legendCategoryName)

    graph.legend.location = "top_right"
    graph.legend.click_policy="mute"
    tab = Panel(child = graph, title = mode)
    return tab 


# List TextData -> 
# Visualizes the data computed (turns into graphs)
def graphData(listTextData):
    listYear, listTTR, listRepetitionPercent, listTitle = getListYears(listTextData), getListTTR(listTextData), getListRepetitionPercent(listTextData), getListTitles(listTextData)
    # print(str(xPosnList)) # TEST
    # print(str(yPosnList)) # TEST
    # print(str(listRepetitionPercent)) # TEST
    # print(str(titleList)) # TEST
    ttrGraph = createTTRGraph(listYear, listTTR, listTitle)
    w1 = widgetbox(Paragraph(text = """Hover over data points to view more information"""))
    w2 = widgetbox(Paragraph(text = """Hover over data points to view more information"""))
    w3 = widgetbox(Paragraph(text = """Click the line/circle tab, legend categories, and hover over data."""))
    wordRepetitionGraph = createWordRepetitionGraph(listYear, listRepetitionPercent, listTitle)
    wtirLineTab, wtirCircleTab = createWTIRGraph(listTextData, listYear, listTitle, "line"), createWTIRGraph(listTextData, listYear, listTitle, "circle")
    tabs = Tabs(tabs = [wtirLineTab, wtirCircleTab])

    graphs, widgets = [ttrGraph, wordRepetitionGraph, tabs], [w1, w2, w3]
    cols = []
    for i in range(0, 3):
        r = row(graphs[i], widgets[i])
        cols.append(r)
    output_file("test.html", title="Test")
    # show(column(ttrGraph, wordRepetitionGraph, wtirGraph))
    show(column(cols))
    return 









### MAIN FUNCTION: ###

def main():
    listTextData = getTextData()
    # printListTextData(listTextData) # TEST
    writeTextDataToFile(listTextData)
    graphData(listTextData)
    return

if __name__ == "__main__":
    main()
