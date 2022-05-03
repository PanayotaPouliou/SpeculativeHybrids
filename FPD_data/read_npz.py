from numpy import load

#data = load('statistics_car_train.npz')
#data = load('statistics_chair_train.npz')
#data = load('statistics_guitar_train.npz')
data = load('/workspaces/SpeculativeHybrids/FPD_data/statistics_building_train.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])