
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import csv

# transforming data from csv to a 2D list  
Meteor_path = "/Users/eduardolopez858/Downloads/Project-1/meteorite-landings.csv"
with open(Meteor_path, mode="r", newline="") as file1:
    MeteorData = list(csv.reader(file1))

'''
# Data Analysis
print("Total number of observations: ", len(MeteorData) - 1)
print("For each observation, we are given the following information about each meteorite: ", MeteorData[0])
# Analyzing each column 
print("The name, Geolocation, reclat, and reclong columns give the location where the meteorite was found, that is, the name of the location and the exact coordinates it was found: ", data[0][0], data[0][9])
print("The id column gives us a unique identification number for each metoerite: ", MeteorData[0][1])
print("The nametype column will either be valid, meaning the meteorite was found as a meteorite object or relict, meaning it was once a meteorite: ", data[0][2])
print("The reclass column gives the type/class of meteorite based on the material it's made out of: ", MeteorData[0][3])
print("The mass column gives the mass of the metoerite: ", MeteorData[0][4])
print("The fall column tells us whether the meteorite was observed falling or had already landed: ", MeteorData[0][5])
print("The year column tells us the year the meteorite was discovered or fell given the evidence: ", MeteorData[0][6])
# Each of these obervations gives us the proper information we need to reach the goal of my probabilistic agent
# Our data is both distributed discretly and continuously:
# Discretly distributed: id, classification, year, fall
# Continuously distributed: name, nametype, mass, reclat, reclong, Geolocation. Although, this data can be categorized in a way that makes it discrete.
'''

# main inference
class BN:
    # constructer (not including headers of data)
    def __init__(self, meteor_data):
        self.meteor_data = meteor_data[1:]

    # categorizing mass of meteorite based on Nasa classification
    def mass_evidence(self, meteor):
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

    # preprocessing for clusteing (only)
    def preprocess(self):
        clust_data = []
        for meteor in self.meteor_data:
            try:
                if "(" in meteor[9] and "," in meteor[9]:
                    lat, lon = meteor[9].strip("()").split(",")
                    lat = float(lat.strip())  
                    lon = float(lon.strip()) 
                else:
                    lat = float(meteor[8])  
                    lon = float(meteor[9])  
                year = int(meteor[6])  
                clust_data.append([lat, lon, year])
            except (ValueError, IndexError):
                continue 
        return np.array(clust_data)

    # getting meteor frequencies based by clustering meteor landings over time
    def frequency_evidence(self):
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

    # inference of next major metoer landing given city
    def likelyhood_given_city(self,city):
        # collecting frequencies of given city
        cluster_counts, data_with_clusters = self.frequency_evidence()
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

    # inference of the danger level of the meteor on city based on mass
    def likelyhood_danger_given_city_mass(self,city):
        mass_data = []
        for meteor in self.meteor_data:
            if city.lower() in meteor[0].lower():
                mass_data.append(self.mass_evidence(meteor))
            if not mass_data:
                return 'None'
        counts = Counter(mass_data)
        highest_likelyhood_mass = max(counts, key=counts.get)
        probability_mass = counts[highest_likelyhood_mass] / len(mass_data)
        print("Most likely mass: ", highest_likelyhood_mass, " With likelihood: ", probability_mass)

# user input (using my model)
def user_interface():
    #input
    city_input = input("Enter city you would like to make an inference on: ")
    # inference
    meteorite_bn = BN(MeteorData)
    prediction_for_city = meteorite_bn.likelyhood_given_city(city_input)
    danger_level = meteorite_bn.likelyhood_danger_given_city_mass(city_input)
    print("There will be:", prediction_for_city, "meteor landings in", city_input)
    print("Specs:", danger_level)

# inference call
user_interface()

