# importing packages
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

# transforming data
path = "/Users/eduardolopez858/Downloads/Projects/Meteorite-Hotspots/meteorite-landings.csv"
with open(path, mode="r", newline="") as file1:
    MeteorData = list(csv.reader(file1))

class Agent():
    # Mass evidence
    def mass_evidence(self, data):
        # relevant column for mass classifications
        extraction = [4]
        # skipping header
        data = [[meteor[i] for i in extraction] for meteor in data[1:]]
        # conversion to np array and removing imputies
        data = np.array([row for row in data if row[0].strip() != ''])
        data = data.astype(float)
        return data
    
    # Time and Location evidence
    def time_location_evidence(self, data):
        # relevant columns for multi-feature clustering (year and coordinate evidence)
        extraction = [6, 7, 8]
        # skipping header
        data = [[meteor[i] for i in extraction] for meteor in data[1:]]
        # conversion to np array and removing imputies
        data = np.array([row for row in data if row[0].strip() != '' and row[1].strip() != '' and row[2].strip() != '']).astype(float)
        # seperating features
        years, lat, lon = data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1), data[:, 2].reshape(-1,1)
        # displaying preprocessed data
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(years, lat, lon, s=3, c='blue', alpha=0.5)
        ax.set_xlabel("Years")
        ax.set_ylabel("Lat")
        ax.set_zlabel("Lon")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        #plt.show()
        return data

    # Frequencies evidence
    def frequencies(self, clean_data):
        # scaling for proper clustering
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clean_data)
        # training model to 3D dataset
        model = DBSCAN(eps=0.3, min_samples=10)
        trained = model.fit(scaled)
        labels = trained.labels_
        # seperating features
        years, lat, lon = clean_data[:, 0].reshape(-1, 1), clean_data[:, 1].reshape(-1, 1), clean_data[:,2].reshape(-1,1)
        # displaying preprocessed data
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(years, lat, lon, s=3, c=labels, cmap='tab10', alpha=0.6)
        ax.set_xlabel("Years")
        ax.set_ylabel("Lat")
        ax.set_zlabel("Lon")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        #plt.show()
        return clean_data, labels

    # Pr(City | Freqs)
    def city_given_freqs(self, x, data, labels):
        # collecting cluster information for inference
        clusters = {}
        total_points = len(labels)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        for label in unique_labels:
            clust_points = data[labels == label]
            if len(clust_points) == 0:
                continue
            n = len(clust_points)
            mu = clust_points.mean(axis=0)
            sig = np.mean(np.linalg.norm(clust_points - mu, axis=1))
            # avoid errors
            if sig == 0 or np.isnan(sig):
                continue
            weight = n / total_points
            # labeling with dict
            clusters[label] = {
                'points': clust_points,
                'centroid': mu,
                'sigma': sig,
                'weight': weight
            }
        # ** Euclidean distance and Gaussian Radial Basis Function kernel**
        probs = {}
        for l, i in clusters.items():
            mu = i['centroid']
            sig = i['sigma']
            weight = i['weight']
            dist = np.sum((x - mu)**2)
            density = np.exp(-dist / (2 * sig**2))
            probs[l] = density * weight
        # normalizing probabilities
        total = sum(probs.values())
        if total == 0:
            return None, 0.0, []
        probs = {l: p / total for l, p in probs.items()}
        result_label = max(probs, key=probs.get)
        result_prob = probs[result_label]
        years = [int(p[0]) for p in clusters[result_label]['points']]
        years_min = min(years)
        years_max = max(years)
        return x, result_label, labels, result_prob, data, years_min, years_max

    # danger level of the meteor
    def danger_given_city_and_mass(self, x, result_prob, label, labels, mass_data, cluster_data, ystart, yend):
        # collecting cluster point mass data
        label_points = cluster_data[labels == label]
        masses_label = [mass_data[p] for p in range(len(cluster_data))]
        masses_average = sum(masses_label) / len(masses_label)
        # categoritzing meteor masses
        if masses_average < 100:
            msg1 = "No Hazard -- Burns Up in Atmosphere"
        elif masses_average > 100 and masses_average < 10000:
            msg1 = "No Hazard -- Bright Fireball and Minor Atmospheric Explosion"
        elif masses_average > 10000 and masses_average < 1000000:
            msg1 = "Minor Risk -- Local Airburst and Possible Window Damage"
        elif masses_average > 1000000 and masses_average < 100000000:
            msg1 = "Local Risk -- Surface Explosion and Major Local damage"
        elif masses_average > 100000000 and masses_average < 1000000000:
            msg1 = "Regional Risk -- Crater Formation and Regional Distruction With Tsunami Potential"
        elif masses_average > 1000000000:
            msg1 = "Global Risk -- Absolute Annihilation and Mass Extinction"
        # final inference
        final_msg = "Location: ", x, "Probability: ", result_prob, "Mass Risk: ", msg1, "Years Range: ", ystart, "-", yend 
        return final_msg

# implementation
Agent1 = Agent()
# user city coordinates along with the year they want to infer
x_new = np.array([2000, 35.00, 139.00])
# pipeline
mass_result = Agent1.mass_evidence(MeteorData)
clean_result = Agent1.time_location_evidence(MeteorData)
data1, labels_result = Agent1.frequencies(clean_result)
point, lab, labs, prob, data2, ymin, ymax = Agent1.city_given_freqs(x_new, data1, labels_result)
final1 = Agent1.danger_given_city_and_mass(point, prob, lab, labs, mass_result, data2, ymin, ymax)
# printing final result
print(final1)
