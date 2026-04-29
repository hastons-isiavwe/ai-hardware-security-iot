import pandas as pd

# File paths for the datasets
file_path1 = "C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\data_1.xlsx"
file_path2 = "C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\data_2.xlsx"
file_path3 = "C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\data_3.xlsx"
file_path4 = "C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\data_4.xlsx"
file_path5 = "C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\HEROdata2.xlsx"

# Reading the datasets
data1 = pd.read_excel(file_path1)
data2 = pd.read_excel(file_path2)
data3 = pd.read_excel(file_path3)
data4 = pd.read_excel(file_path4)
data5 = pd.read_excel(file_path5)

# Displaying a preview of each dataset
print("Preview of data1:")
print(data1.head())

print("\nPreview of data2:")
print(data2.head())

print("\nPreview of data3:")
print(data3.head())

print("\nPreview of data4:")
print(data4.head())

print("\nPreview of data5:")
print(data5.head())
