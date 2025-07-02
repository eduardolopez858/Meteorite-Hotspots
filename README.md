# Meteorite Hotspots
## Abstract
#### Using PEAS (Performance measure, Environment, Actuators, Sensors)
My agent's performance measure will be meteorite detection, that is, to find fallen meteorites for the purpose of research and worldwide safely prediction by modeling and inferencing how frequent these meteorites will fall and be considered dangerous based on historical data. My agent will have a fully observable environment of planet earth, given geographical locations, a timeline, the masses of the meteorites, and more. My agent's actuators will be predictive outcomes on meteorite landings such as predicting the next meteor shower based on the given evidence(clustering data and finding patterns depending location -> lat, lon), predictive outcomes on specific meteorite landing locations(which depends on frequencies on a given location), and the predictive danger level of a meteorite(which depends on mass and frequencies), all which can be modeled in a probabilistic way using a Bayesian Network. Lastly, my agent will have sensors of gps(world map), time collection if needed, and meteorite mass measurement.

## My agent's setup
Using the chosen dataset, the main variables my model will be utilizing the mass, location, and landings timeline of the metoerites. The reason being my model's purpose is to infer and predict the likelyhood of a meteorite falling on a given location, as well as additionally infering they're danger level, that is when given their mass. 

The variables start off with mass, location , and timeline evidence. These are the baseline evidence variables the model depends on for future computations.

The mass variable is represented as a function that classifies the masses of the meteorites (using Nasa classifications) that will be used in future computations.
```ruby
 def mass(self, meteor):
        try:
            mass_kg = int(float(meteor[4]))
            if mass_kg <= 1:
                return 'Harmless. Most likely burn up in the atmosphere'
            if mass_kg > 1 and mass_kg <= 10:
                return 'Some property damage'
            if mass_kg > 10 and mass_kg <= 100:
                return "More localized damage, may cause injuries"
            if mass_kg > 100 and mass_kg <= 10000:
                return "Huge shockwave. May destory buildings"
            if mass_kg > 10000 and mass_kg <= 1000000:
                return "Regional threat. May destory entire cities and even cause tsunamis"
            if mass_kg > 1000000:
                return "Global threat. Mass extinctions"
        except (ValueError, IndexError, TypeError):
            return 'None'
```

On the other hand, the location and timeline evidence variables will need to be preproccessed for the main computation of the model.

```ruby
def pre(meteor_data):
    # columns 0 and 6 are the only relevant in this computation
    extraction = [0,6]
    clust_data = [[meteor[i] for i in extraction] for meteor in meteor_data]
    clust_data = clust_data[1:]

    # converting dataset into a clean NumPy array
    arrayPre = np.array(clust_data)
    cleaned_array = np.array([row for row in arrayPre if row[1].strip() != ''])

    # displaying relevant dataset
    years = cleaned_array[:, 1].astype(float)
    cities = cleaned_array[:, 0]
    plt.figure(figsize=(8, 8))
    plt.scatter(years, cities, s=1)
    x_min = years.min()
    x_max = years.max()
    plt.ylabel("Cities")
    plt.xlabel(f"Years {int(x_min)} - {int(x_max)}")
    plt.yticks([])
    plt.show()  
```
We only need preproccess these two variables (cities and time) because the next variable of the model (frequencies) uses them to classify the meteorite landings as clusters, that is, transforming the data into only location and time numerical features for the DBSCAN algorithm. It's important for this model as we can use the time and location variables of the metoerite landings to find patterns by clustering these landings based on their location and time and categorize them as metoerite hotspots. Although, before we do that, we need to first use the preprocessing function above, giving us the following scattar plot:
![](https://github.com/eduardolopez858/Meteorite-Hotspots/blob/main/Model1.png) 





model visualization coming soon*

```ruby
def frequencies(self):
        data = self.preprocess()
        if data.size == 0:
            return 'No valid clustering data'
        dbscan = DBSCAN(eps=5, min_samples=2) 
        clusters = dbscan.fit_predict(data[:, :2]) 
        cluster_counts = Counter(clusters)
        data_with_clusters = np.column_stack((data, clusters))
        if -1 in cluster_counts:
            del cluster_counts[-1]
        return cluster_counts, data_with_clusters
```

With this, we have now created a frequencies variable that can now be used to infer the likleyhood of a meteorite landing on given location.

```ruby
def likelyhood_given_city(self,city):
        cluster_counts, data_with_clusters = self.frequencies()
        if not cluster_counts or data_with_clusters.size == 0:
            return 'None'
        city_cluster = None
        for meteor in self.meteor_data:
            if city.lower() in meteor[0].lower():
                try:
                    lat = float(meteor[8])
                    lon = float(meteor[9])
                    for point in data_with_clusters:
                        if point[0] == lat and point[1] == lon:
                            city_cluster = point[3]
                            break
                except (ValueError, IndexError):
                    continue
    
        if city_cluster is None or city_cluster == -1:
            return 0
        total_landings = sum(cluster_counts.values())
        city_landings = cluster_counts.get(city_cluster, 0)
        if total_landings == 0:
            return 0
        return city_landings / total_landings
```

Finally, we can have fun with the model and infer the danger level of the meteorites using uniform distribution that can tell the user the danger level of the meteorite based on frequency and mass(e.g "Global threat. Mass extinctions"). 

```ruby
def likelyhood_danger_given_city_mass(self,city):
        mass_data = []
        for meteor in self.meteor_data:
            if city.lower() in meteor[0].lower():
                mass_data.append(self.mass(meteor))
            if not mass_data:
                return 'None'
        counts = Counter(mass_data)
        highest_likelyhood_mass = max(counts, key=counts.get)
        probability_mass = counts[highest_likelyhood_mass] / len(mass_data)
        print("Most likely mass: ", highest_likelyhood_mass, " With likelihood: ", probability_mass)
```
For testing purposes, I have created a temporary user interface. Front end user interface is still under development.

```ruby
def user_interface():
    #input
    city_input = input("Enter city you would like to make an inference on: ")
    # inference
    meteorite_bn = BN(MeteorData)
    prediction_for_city = meteorite_bn.likelyhood_given_city(city_input)
    danger_level = meteorite_bn.likelyhood_danger_given_city_mass(city_input)
    print("There will be:", prediction_for_city, "meteor landings in", city_input)
    print("Specs:", danger_level)
```

## Model Structure:   
![]()
 
## Library sources:
https://docs.python.org/3/library/collections.html   
https://docs.python.org/3/library/csv.html   
https://scikit-learn.org/stable/   
https://numpy.org/   
https://networkx.org/   
https://matplotlib.org/   
