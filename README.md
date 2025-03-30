# Meteorite Hotspots
### Abstract: 
#### Performance measure:
My agent's performance measure will be meteorite detection, that is, to find fallen meteorites for the purpose of research and worldwide safely prediction by modeling and inferencing how frequent these meteorites will fall and be considered dangerous based on historical data. 
#### Environment: 
My agent will have a fully observable environment of planet earth, given geographical locations, a timeline, the masses of the meteorites, and more. 
#### Actuators:
My agent's actuators will be predictive outcomes on meteorite landings such as predicting the next meteor shower based on the given evidence(clustering data and finding patterns depending location -> lat, lon), predictive outcomes on specific meteorite landing locations(which depends on frequencies on a given location), and the predictive danger level of a meteorite(which depends on mass and frequencies), all which can be modeled in a probabilistic way using a Bayesian Network. 
#### Sensors:
Lastly, my agent will have sensors of gps(world map), time collection if needed, and meteorite mass measurement.

#### My agent's setup:
Using the dataset I chose, the main variables my model will be utilizing the mass, location, and landings timeline of the metoerites. The reason being the goal on my model is to infer and predict the likelyhood of a meteor falling on a given location, as well as additionally infering they're danger level given their mass. 

The variables start off with mass, location , and timeline evidence. These are the baseline evidence variables the model depends on to use in future computations. The next variable uses evidence of time and location to model the meteorite landings as clusters(using a preprocessing function to filter out the data into only location and time (2D) and integers/floats for the DBSCAN algorithm). This is important for this type of model as we can use the history (time and location) of the metoerite landings to find patterns by clustering these landings based on their location and time and categorize them as metoerite hotspots. With this, we have now created a frequencies variable that can be used to estimate how often these meteorites will fall in groups of clusters over time (in the next upcoming years based timeline span) for a given location, that is, our first main computation and second to last variable that depends on the frequencies variable. Finally, we can have fun with the model and infer the danger level of the meteorites using basic probability theory(uniform distribution) that can tell the user how dangerous the meteorites can be based on frequency and mass(e.g "Global threat. Mass extinctions").







This model uses 4 different types of libraries for assistance. It uses csv to read and transform the csv dataset file into a 2D array for better implementation of python code. It uses numpy in case we need to transform any lists into array for numerical results(e.g input array for DBSCAN). It also uses the Counter algorithm from the collections library for easier use of counting occurrences of a certain item in a data structure. In this case, I used it to efficiency give the mass of the fallen meteorites with the most occurrences to uniformly predict the danger level of said meteorites, as well as for cluster counting.Lastly, from the sklearn library, specifically in the clustering module, I used the DBSCAN algorithm to utilize the preprocessed data of time and location to cluster/group metoerite landings and consider them as landing hotspots. I used this algorithm as my main structure of inference because like I previously stated, this type of dataset and inference requires pattern detection of the meteorite landings with locations/time uses as factors because the goal of the model is to predict these type of regional hotspots. I considered using a different algorithm from sklearn, that is, K-means(which I learned from cse 151a to be great for this type of data) but DBSCAN is more effective in removing single noise meteorite landings that would not be useful for our clusterings. The algorithm has the minimum considered sample size of a cluster of meteorite landings set to 2 but in future training, I want the user input to decide the sample size for a better and more accurate inference given what they want to know. Same with the eps value, it would largely depend on the accuracy of the given dataset.

### Conclusion:
Results: the lower the numerical results, the likelyhood decreases of a landing occuring in the specifc city. 
My model can improve in various aspects. To start, the chosen dataset was not as accurate as I thought it would and therefore my model lacked a lot of extra variables that could have helped create a more accurate inference on the same goal. For example, the speeds of the meteorites (if recorded) could have been more useful in inferencing the danger level of the meteorite. Given the landing's regions of the metoerite could have made the inference more interesting for the user as they would not be limited to single cities now. The preprocessing function also needs some work becuase of the way the dataset was structured. It lacks some accuracy which can effect the performance of the DBSCAN algorithm. Lastly, better adjustments for the DBSCAN algorithm parameters as previously mentioned could have also improved the inference of the model based on what the user wanted to look for, and any kind of biases that they would have considered as well(like minimum cluster size). 



### Links to libraries:
https://docs.python.org/3/library/collections.html
https://docs.python.org/3/library/csv.html 
https://scikit-learn.org/stable/
https://numpy.org/
https://networkx.org/

