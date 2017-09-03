### Wrangle OpenStreetMap Data with SQL

In this project, we will choose any area of the world in https://www.openstreetmap.org and use data munging techniques, such as assessing the quality of the data for validity, accuracy, completeness, consistency and uniformity, to clean the OpenStreetMap data. Finally, we will use SQL as the data schema to complete the project.

#### Choose Map Area
Choose any area of the world and download a XML OSM dataset. Use one of following methods of downloading a dataset:

- Download a preselected metro area from [Map Zen](https://mapzen.com/data/metro-extracts/).
- Use the [Overpass API](http://overpass-api.de/query_form.html) to download a custom square area. Explanation of the syntax can found in the [wiki](http://wiki.openstreetmap.org/wiki/Overpass_API). In general you will want to use the following query:(node(minimum_latitude, minimum_longitude, maximum_latitude, maximum_longitude);<;);out meta; e.g. (node(51.249,7.148,51.251,7.152);<;);out meta; the meta option is included so the elements contain timestamp and user information. You can use the Open Street Map [Export Tool](http://www.openstreetmap.org/export#map=5/42.618/-7.559) to find the coordinates of your bounding box. *Note: You will not be able to use the Export Tool to actually download the data, the area required for this project is too large.**

We have choosen Map Area as [San Diego, California, USA](https://mapzen.com/data/metro-extracts/metro/san-diego_california/).

#### Take sample from large OSM data
Use [this](https://github.com/gauravansal/Wrangle-OpenStreetMap-Data/blob/master/Take%20systematic%20sample%20from%20large%20OSM%20data.md) code to take a systematic sample of elements from your original OSM region.

#### Process the Dataset
As we unravel the data, we will take note of problems encountered along the way as well as issues with the dataset.

In this project we have used SQL rather than MongoDB, so we will be working with CSV file rather than JSON file. We will perform data auditing & cleaning when we convert the XML into CSV format.

*We won't change the original data file, only the data that we plan on inserting into our database. This is where our organization will pay off, since we can just import the update functions from our auditing scripts into the cleaning and conversion script.*

Then import the cleaned .csv file(s) into a SQL database using this [schema](https://gist.github.com/swwelch/f1144229848b407e0a5d13fcb7fbbd6f) or a custom schema of your choice.

#### Explore the Database
After building our local database we’ll explore the data by running queries.

#### Note -
- When opening ipython notebook on github, github might fail to load it correctly on github viewer, please use Juypter NBViewer: http://nbviewer.jupyter.org/github/gauravansal/Wrangle-OpenStreetMap-Data/blob/master/Wrangle%20OpenStreetMap%20Data.ipynb to see the full ipython notebook for the project.
- OSM posted in this github is a small portion of large MAPZEN OSM data(around 300 MB), while this sample OSM is about 8 MB in size.



