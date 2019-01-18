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

## Running the Program
Create a directory that includes metrics.py. Within that directory create another directory where you can store text samples (or download the TEXT_SAMPLES folder from this repository). Format all text samples as:

```
Date_CamelCaseTitle_PersonInitials.txt
```

To run in terminal (make sure you're in the right directory):
```Bash
python metrics.py
```

The program will prompt you for a text file to write generated data to. It will create an html file in the directory sotring graph data.


## Interpreting Results
### Type/token Ratio Graph
![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/TTRGraph.png) 

This graph displays the TTR for each text sample, plotted over time. If the line of best fit has a negative slope, this suggests that vocabulary declines in the writing samples over time. The decline implies linguistic regression, which may be a sign of dementia. 

The example graph shown above displays text analysis of Agatha Christie's writings. Her vocabulary measured by the TTR follows a decreasing trend, which suggests linguistic decline. Compare with results generated by the [paper](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/Screen%20Shot%202019-01-18%20at%2012.55.10%20PM.png) (for reference, the paper compares writings of 3 authors: Agatha Christie, who is suspected to have had dementia; P.D. James, who did not have dementia; and Iris Murdoch, who had Alzheimer's).

### Word Repetition Percent Graph 
![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/RepetitionGraph.png)

This graph displays the word repetition percent for each text sample, plotted over time. If the line of best fit has a positive slope, this suggests that the person tends to repeat more words over time. The increase implies linguistic decline, which may be a sign of dementia. 

The example graph shows the repetition percent of Agatha Christie's writings over time. Her tendency to repeat more words and her declining word diversity suggests linguistic decline. Compare with results generated by the [paper](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/Screen%20Shot%202019-01-18%20at%201.54.00%20PM.png).

### Word Type Introduction Rate Graphs
![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/AllLinesGraph.png) ![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/AllCircleGraph.png)

These graphs display the word types counted at every 10,000th interval for each book.

Clicking on the legend to view WTIR by category will more clearly reveal linguistic decline, if any. Lower word types introduced or slower rates of introduction is indicative of linguistic decline.

![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/LinesLegendGraph.png) ![](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/CircleLegendGraph.png)

These graphs above show the WTIR of Agatha Christie's writings. When viewed by categories of "earlier works" and "later works," lower word diversity and lower word introduction rates can be seen in the "later works" category. Compare with results generated by the paper: [James Graph](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/Screen%20Shot%202019-01-18%20at%202.11.15%20PM.png),
[Christie Graph](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/Screen%20Shot%202019-01-18%20at%202.11.23%20PM.png),
[Murdoch Graph](https://github.com/JoyD424/Dementia_Detection/blob/master/Images/Screen%20Shot%202019-01-18%20at%202.11.30%20PM.png).






