### Data Visualization with D3.js
In this project, we will create an explanatory data visualization from a data set that communicates a clear finding or that highlights relationships or patterns in a data set. Our work should be a reflection of the theory and practice of data visualization.

Dataset used in this project is [Loan Data from Prosper](https://www.google.com/url?q=https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv&sa=D&ust=1504419324115000&usg=AFQjCNFbQ446hXT7ODuqheLQJVigvu8XLQ).

Since, it is large dataset(which contains 81 variables), we have used R tool to perform data exploration,wrangling and cleaning to produce a smaller dataset to be used in this project. Smaller dataset(which contains 7 variables only) will help render the data visualization quickly and easily over web and tells a story highlighting trends and patterns in the data.

[Smaller dataset](https://github.com/gauravansal/Data-Visualization-with-D3.js/blob/master/bar_chart_prosper_selected_variables_outliers_removed.tsv) contains selected variables and outliers removed in a TSV file.

#### Summary
Prosper Rating of a borrower depends upon the number of factors. And based upon the combination of Prosper Rating and values of the factors, a Borrower Rate is calculated for a borrower. Normally, higher the Prosper Rating, lower the Borrower Rate and vice-versa. Prosper rating and Borrower Rate are negatively correlated with each other. So, as Prosper rating of a borrower increases, Borrower Rate for the borrower lowers down and vice-versa. Rating 1 is the lowest and Rating 7 is the best. Prosper helps lends money to borrower for various Tenure. In this visualization, relationships between Prosper Rating & factors is shown. There is a separate bar chart for each of the factor with respect to Prosper Rating. 

Visualizaion can be viewed at http://bl.ocks.org/gauravansal/raw/0001a9f402cf7b54b961548816f2cc27/

***
#### Design
Chart Type - is Bar chart. I chose bar chart as length of bar would determine of average value of a factor. User can compare the length of the bars for a factor for different Ratings and can derive analysis to come to a conclusion if that factor is positively or negatively correlated with Prosper Rating.

Visual Encodings - 
1) Length / Size of a bar - encodes the average value of a particular factor for a particular Rating. 

2) Direction / Y-axis - has been used to represent all the factors and the scale used in Y-axis is linear. Y-axis represents a single factor at one time based on the factor used or selected. Y-axis values changes depending upon the factor used / selected.

3) Direction / X-axis - has been used for Propser Rating and the scale is Ordinal. One can easily compare the length of the bars for each of the Rating at the same time and the user can have an insight regarding the relationship between a particular factor and Prosper Rating.  

Legends - Legends for Prosper rating and Factors has not been used as both of these are mentioned on the X-axis, Y-axis and header itself.

Layout - There is different bar chart for each of the factor. Each of the chart can be seen by selecting the appropriate factor from the dropdown menu. 

Here, Length of a bar represents average value of a factor for that particular Prosper Rating. In each bar chart for a particular factor, you would see increase and decrease in the length of bars as we move from Rating 1 to Rating 7 denoting increase or decrease in the average value of that factor as we move from Rating 1 to Rating 7.Below are the key take aways of this visualization which determines how each factor affect Prosper Rating - 

1) Credit Score, Monthly Income, Revolving Credit Balance & Open Credit Lines are positively correlated with Prosper Rating, since as we move from Rating 1 to Rating 7, length of bars increases, i.e average value of factor increases. 

2) Debt to Income Ratio & Inquiries Last 6 Months are negatively correlated with Prosper Rating, since as we move from Rating 1 to Rating 7, length of bars decreases, i.e average value of factor decreases.

3) People with Higher Rating would have high Credit Score, high Monthly Income, high Revolving Credit Balance, high number of Open Credit Lines, low Debt to Income Ratio & low number of Inquiries within Last 6 Months.

4) Similarly, people with Lower Rating would have lower Credit score, low Monthly Income, low Revolving Credit Balance, low number of Open Credit Lines, high Debt to Income Ratio & high number of Inquiries within Last 6 Months.

 A user can select a factor from the dropdown menu in order to have the bar chart for that particular selection. Since, the factor is changed, the Y-axis values changes accordingly depending upon the factor selected and a bar chart with different bar lengths is produced.
***
#### Note - 
To work on your data visualization, you will need to start a local web server on your computer. To learn more about why you need to start a local web server and ways of setting up a local web server, please read [Setting Up A Local Web Server](http://chimera.labs.oreilly.com/books/1230000000345/ch04.html#_setting_up_a_web_server) from Scott Murray's book, Interactive Data Visualization for the Web.

#### Resources - 

1) https://www.w3schools.com/
2) https://stackoverflow.com/
3) http://jsfiddle.net/
4) http://www.d3noob.org/
5) http://bl.ocks.org/
6) https://www.dashingd3js.com/
