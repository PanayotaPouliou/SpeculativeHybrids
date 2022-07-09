from pathlib import Path
import numpy as np
import torch
import pandas as pd
from sqlalchemy import column, false
import os
import csv

def modify_pc_0(pc_0_path):
    def read_pts(filename):
        return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

    def distance(x, y):
        if x >= y:
            result = x - y
        else:
            result = y - x
        return result


    dir = pc_0_path
    i=0

    for filename in os.listdir(dir):
        f = os.path.join(dir,filename)

        n_array = read_pts(f)

        df = pd.DataFrame(n_array, columns = ['Column_A','Column_B','Column_C', 'Column_D', 'Column_E', 'Column_F', 'Column_X'])
        #print(df)
        #print(len(df))

        labels = []
        for i in range(0,len(df)):
            a = df['Column_D'].values[i]
            b = df['Column_E'].values[i]
            c = df['Column_F'].values[i]

            largest = 0

            if a > b and a > c:
                largest = 1
            if b > a and b > c:
                largest = 2
            if c > a and c > b:
                largest = 3

            labels.append(largest)

        df['Labels'] = labels
        
        for i in range(0,len(df)):
            label = df['Labels'].values[i]
            x = df['Column_A'].values[i]
            y = df['Column_B'].values[i]
            z = df['Column_C'].values[i]

            ##PARAMETERS AND RULES FOR MAPPING regarding building width - length
            
            for s in range(0,len(df)):
                xs = df['Column_A'].values[s]
                distanceX = distance(x,xs)
                limitX = 10 #The limit is a variable chosen by the user (depending on the field) maximoum x distance

                if distanceX > limitX: 
                    w = distance/2
                    if x>=xs:
                        if x>=0:
                            cent = x - w
                        df.replace(x, cent+(limitX/2))
                        df.replace(xs, cent-(limitX/2))
                    else:
                        if xs>=0:
                            cent = xs - w
                        df.replace(x, cent-(limitX/2))
                        df.replace(xs, cent+(limitX/2))


                ys = df['Column_B'].values[s]
                distanceY = distance(y,ys)
                limitY = 15 #The limit is a variable chosen by the user (depending on the field) maximoum y distance

                if distanceY > limitY: 
                    w = distance/2
                    if y>=ys:
                        if y>=0:
                            cent = y - w
                        else:
                            cent = w + y
                        df.replace(y, cent+(limitY/2))
                        df.replace(ys, cent-(limitY/2))
                    else:
                        if y>=0:
                            cent = ys - w
                        else:
                            cent = w + ys
                        
                        df.replace(y, cent-(limitY/2))
                        df.replace(ys, cent+(limitY/2))

            ##PARAMETERS AND RULES FOR MAPPING regarding Facade height
            if label == 1:
                limitZ = 8 #The limit is a variable chosen by the user (depending on the field) maximoum z distance from wall points
                if z>limitZ:
                    df.replace(z, limitZ)

            ##PARAMETERS AND RULES FOR MAPPING regarding Maximum B height height
            if label == 2:
                limitZmin = 8 #The limit is a variable chosen by the user (depending on the field) maximoum z distance from wall points (limitZ)
                limitZmax = 15 #The limit is a variable chosen by the user (depending on the field) maximoum z distance (total buinding height)
                if z<limitZmin:
                    df.replace(z, limitZmin)
                if z>limitZmax:
                    df.replace(z, limitZmax)
                                    
        df = df.drop(['Labels'], axis=1)

        print(df)

        #Save as .pts
        df.to_csv(pc_0_path, index=False, header=False)
        i=i+1

        with pc_0_path.open("r") as f:
            pc_0 = np.loadtxt(f)
    return pc_0

def experience(self):
    model = self.model
    # load z and structure point cloud
    z_path = Path("./z.npy")
    pc_0_path = Path("./pc_0.pts")
    z,pc_0 = None,None
    with z_path.open("rb") as f:
        z = torch.load(f).view(1,-1)
    #with pc_0_path.open("r") as f:
        #pc_0 = np.loadtxt(f)
    
    # Modify the structure point cloud
    pc_0 = modify_pc_0(pc_0_path)
    # To GPU
    z = z.cuda()
    pc_0 = torch.from_numpy(pc_0).float().view(1,-1,7).cuda()
    
    f_pc_1 = model.gen_from_given_z_and_pc0(z,pc_0)

    # Save generated point cloud
    pc_1_path = Path("./pc_1.pts")
    with pc_1_path.open("w") as f:
        np.savetxt(f,f_pc_1[0].cpu().numpy())
    