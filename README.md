# Dementia_Detection

## Description
This project analyzes writing samples over a span of decades for markers of linguistic decline commonly present in cases of dementia.

This project draws from the paper 
["Longitudinal Detection of Dementia Through Lexical and Syntactic Changes in Writing" (Le, Xuan)](https://academic.oup.com/dsh/article/26/4/435/1052059). The three metrics that the paper found to have produced statistically significant results—type/token ratio, word type introduction rate, and word repetition percent—are implemented in this program.

The program will return a text file (.txt) with the metrics generated and a static HTML file (.html) that displays visualizations (graphs) for data generated

### Metric Definitions
* Type/token ratio: division of the types (total number of _different_ words) by lemmatized (the stem for variants/inflections of the same word) tokens (total number of words). This is essentially a measure of lexical diversity (vocabulary size).
* Word type introduction rate: calculates the word types introduced so far at every 10,000th token interval.
* Word repetition percent: percent of lemmatized content words (stem words that are either verbs, nouns, adjectives, or adverbs) in the writing sample


## Interpreting Results


