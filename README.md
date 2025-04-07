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

The variables start off with mass, location , and timeline evidence. These are the baseline evidence variables the model depends on to use in future computations. 

The next variable uses evidence of time and location to model the meteorite landings as clusters(using a preprocessing function to filter out the data into only location and time (2D) and integers/floats for the DBSCAN algorithm). This is important for this type of model as we can use the history (time and location) of the metoerite landings to find patterns by clustering these landings based on their location and time and categorize them as metoerite hotspots.

With this, we have now created a frequencies variable that can be used to estimate how often these meteorites will fall in groups of clusters over time (in the next upcoming years based timeline span) for a given location, that is, our first main computation and second to last variable that depends on the frequencies variable. Finally, we can have fun with the model and infer the danger level of the meteorites using basic probability theory(uniform distribution) that can tell the user how dangerous the meteorites can be based on frequency and mass(e.g "Global threat. Mass extinctions").   

<img width="792" alt="Image" src="https://github.com/user-attachments/assets/0440e161-712c-4424-a152-724c9a60ab84" />   
 
#### Libraries:
https://docs.python.org/3/library/collections.html   
https://docs.python.org/3/library/csv.html   
https://scikit-learn.org/stable/   
https://numpy.org/   
https://networkx.org/   
https://matplotlib.org/   
