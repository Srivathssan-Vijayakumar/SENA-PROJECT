#import statements
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Read csv files
df = pd.read_csv('Edges.csv')
df.head()
df.columns

#Create Graph From edge list
G = nx.from_pandas_edgelist(df,source='Source',target='Target')
plt.figure(figsize=(20,20))
nx.draw(G)

#Plot histogram
plt.hist([v for k,v in nx.degree(G)])

#print average clustering co efficient
nx.cluster.average_clustering(G)

#print diameter - infinite diameter
try:
    nx.diameter(G)
except Exception as e:
    print('Infinite Diameter')
# len(G)

#Conversion of graph to edge matrix for clustering
def graph_to_edge_matrix(G):
    edge_mat = np.zeros((len(G),len(G)),dtype=int)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node-1][neighbor-1]=1
        edge_mat[node-1][node-1]=1
    return edge_mat

#Read csv for cluster analyses
df = pd.read_csv('Placement_Data_Full_Class.csv')
df.head()
df.columns

#Create custom dataset
dataset = {
    'Sl_no':[],
    'Branch And Specialization':[],
    'Placement Status':[],
    'Salary':[]
}
for index,row in df.iterrows():
    dataset['Sl_no'].append(row['sl_no'])
    dataset['Branch And Specialization'].append(row['degree_t']+' '+row['specialisation'])
    dataset['Placement Status'].append(row['status'])
    dataset['Salary'].append(row['salary'])


len(df)
len(dataset['Salary'])

#Dataset analysis
df.info()

#Statistical analysis
df.describe()

#null values checking
df.isna().sum()

#Creation of custom dataset
Dataframe = pd.DataFrame(dataset)
Dataframe.head()
Dataframe.isnull().sum()

#Filling of null values
Dataframe.fillna(value=0.0,inplace=True)
Dataframe.head()

#Printing count of students in each brnch and specialisation
Dataframe['Branch And Specialization'].value_counts()

#Count of students who are placed in  each department
lst={
    'Sci&Tech Mkt&HR':list(Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&HR']['Placement Status'].value_counts()),
    'Sci&Tech Mkt&Fin':list(Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&Fin']['Placement Status'].value_counts()),
    'Comm&Mgmt Mkt&Fin':list(Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&Fin']['Placement Status'].value_counts()),
    'Comm&Mgmt Mkt&HR':list(Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&HR']['Placement Status'].value_counts()),
    'Others Mkt&HR':list(Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&HR']['Placement Status'].value_counts()),
    'Others Mkt&Fin':list(Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&Fin']['Placement Status'].value_counts()),
}
lst
avg_salary = []
# Average Salary of Sci&Tech Mkt&HR
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&HR']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&HR'])).round(2))
# Average Salary of Comm&Mgmt Mkt&Fin
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&Fin']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&Fin'])).round(2))
# Average Salary of Comm&Mgmt Mkt&HR
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&HR']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Comm&Mgmt Mkt&HR'])).round(2))
# Average Salary of Sci&Tech Mkt&Fin
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&Fin']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Sci&Tech Mkt&Fin'])).round(2))
# Average Salary of Others Mkt&HR
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&HR']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&HR'])).round(2))

# 
# Average Salary of Others Mkt&Fin
avg_salary.append((Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&Fin']['Salary'].sum()/len(Dataframe[Dataframe['Branch And Specialization']=='Others Mkt&Fin'])).round(2))

#avg_salary

#plotting dept vs placed students count 
bxaxis = ['Sci&Tech Mkt&HR','Sci&Tech Mkt&Fin','Comm&Mgmt Mkt&Fin','Comm&Mgmt Mkt&HR','Others Mkt&HR','Others Mkt&Fin']
byaxis = [lst[bxaxis[_]][1] for _ in range(len(lst))]

plt.figure(figsize=(14,5))
plt.bar(bxaxis, byaxis, color='g')
plt.title("Number of Placed Students")
plt.xlabel("Department")
plt.ylabel("Number of Students")
plt.show()

#plotting dept vs average salary
gxaxis = ['Sci&Tech Mkt&HR','Comm&Mgmt Mkt&Fin','Comm&Mgmt Mkt&HR','Sci&Tech Mkt&Fin','Others Mkt&HR','Others Mkt&Fin']
plt.figure(figsize=(14,5))
plt.bar(gxaxis, avg_salary, color='b')
plt.title("Student's Average Salary")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.show()

# Final Analysis
print("Hence from this analysis we can know that " + gxaxis[avg_salary.index(max(avg_salary))] + " has high Average Salary")
print("Hence from this analysis we can know that " + bxaxis[byaxis.index(max(byaxis))] + " has high Highest Placement")


