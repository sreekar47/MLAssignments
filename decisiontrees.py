import pandas as pd
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Data File")
args = parser.parse_args()
file= args.data

features = pd.read_csv(file,header=None)

#naming the features starting from 0
features.columns = ["att"+ str(x) for x in range(len(features.iloc[1,:]))]

#to calculate the total entropy of the features
def totalentropy(trainingdata):
    total_row = trainingdata.shape[0] #the total size of the dataset
    total_entr = 0
    labels = features.iloc[:,-1].unique() #no of lable classes in the given dataset
    logbase = labels.shape[0]
    #if total_row!=0:
    for class_label in labels: 
      classcount = trainingdata[class_label == trainingdata.iloc[:,-1]].shape[0] #number of the class
      if classcount!=0:
        prob = classcount/total_row
        entropytotal = - (prob)*(math.log(prob, logbase)) #entropy of the class
        total_entr += entropytotal #adding the class entropy to the total entropy of the dataset    
    return total_entr

#to calculate the individual class entropy in each column
def attributesentropy(column, features, attribute_value):
      attribute_value_data = features[features[column] == attribute_value] #filtering rows with that attribute_value
      attribute_value_count = attribute_value_data.shape[0]
      attributesentropy = totalentropy(attribute_value_data) #calculating entropy for the attribute value
      return attributesentropy


#to find the split information(information gain and best split) for the decision tree
def splitinformation(features):
   best_split = None
   info_gain = 0 
   class_entropy = totalentropy(features) 
   total_row = features.shape[0]
   for col in features.columns:
    if col != features.columns[-1]:
      attribute_value_list = features[col].unique()
      attribute_info = 0
      for attribute_value in attribute_value_list:
        attribute_value_count = features[features[col] == attribute_value].shape[0]
        value_entropy = attributesentropy(col, features, attribute_value)
        attribute_value_probability = attribute_value_count/total_row
        attribute_info += attribute_value_probability * value_entropy #calculating information of the attribute value
        cur_info_gain = class_entropy - attribute_info   
    if cur_info_gain > info_gain:
          best_split = col
          info_gain = cur_info_gain        
   return info_gain, best_split


def calculate_sub_data(data, attr, attribute_value):
 subDataset = data[data[attr] == attribute_value]
 return subDataset

#to print the values
def ID3(data, depth=1): 
  initial_depth = 0
  information_gain, best_attr = splitinformation(data)
  attribute_list = data[best_attr].unique()
  for attribute_value in attribute_list: 
    x = attributesentropy(best_attr, data, attribute_value)
    subData = calculate_sub_data(data, best_attr, attribute_value)
    if(x!=0 and information_gain!=0):
      print(f'{depth},{best_attr}={attribute_value},{x},no_leaf ')
      ID3(subData, depth=depth+1)
    else:
      print(f'{depth},{best_attr}={attribute_value},{x},{subData.iloc[0,-1]}')

print(f'0,root,{totalentropy(features)},no_leaf')
ID3(features)