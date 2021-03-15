"""
This script regress the site and framewise displacement (FD) from the ABIDE data.
It needs the matrix of sites and FD by subject. It saves the subjects' matrices 
in MATLAB format (.mat).
"""

from __future__ import print_function
import os
from os import listdir
from os.path import isfile, join
import scipy.io
import numpy as np
import csv
from sklearn import linear_model
import scipy.io

def load_adj_matrix_from_mat(fname, var_name='Adj'): 
    """
    Load a correlation/adjacency matrix using .mat file format

    Parameters
    ----------
    fname : str
        path to the .mat file containing the  matrix
    var_name : str
        the variable name of the matrix
    """
    assert fname[-4:] == ".mat", "Trying to load incorrect file format"
    adjMatrix = load_mat(fname, squeeze_me=False)[var_name]
    adjMatrix = np.triu(adjMatrix, 1)
    return adjMatrix
    
def load_mat(fname, squeeze_me=False):
    """
    Loads an arbitrary .mat file.

    Parameters
    ----------
    fname : str
        path to the .mat file
    squeeze_me : bool
        whether or not to squeeze additional
        matrix dimensions used by matlab.

    Returns
    -------
    data_dict : dict
        a dictionary with the 'mat' variable names as keys,
        and the actual matlab matrices/structs etc. as values

    Limited support is also available for HDF-matlab files.
    """
    try:
        data_dict = scipy.io.loadmat(fname, squeeze_me=squeeze_me)
    except NotImplementedError as e:
        if e.message == "Please use HDF reader for matlab v7.3 files":
            import h5py
            data = h5py.File(fname, "r")
            data_dict = {}
            for key in data.keys():
                if squeeze_me:
                    try:
                        data_dict[key] = np.squeeze(data[key])
                    except:
                        data_dict[key] = data[key]
                else:
                    data_dict[key] = data[key]
        else:
            raise e
    return data_dict

def read_all_subjects(Smoothing, folder_g1,kind):
    """
    Organizes adjacency matrices in a folder as part of a group.  

    Parameters
    ----------
    Smoothing : list 
        of Smoothing levels, usually the same folder names used to store the smoothing levels
    Threshold : str
        Threshold level used for the matrix, usually the same str as the folder name
    folder_g1 : str
        folder path where the adjacency matrices are

    Returns
    -------
    smooth : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and subject as sencond keys. The actual matlab matrices/structs etc. as values

    """
    #Verified process :)
    #Organize the adjacency matrices in dictionaries for group (ASD or TC), for all smoothing levels
    smooth1=dict()
    for idx,smooth in enumerate(Smoothing):
        folder=folder_g1+Smoothing[idx]
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        
        files=[x for x in files if kind in x ]
        group=dict()
        for file in files:
            fname=file
            adjMat=load_adj_matrix_from_mat(folder+fname) #this is a numpy object, so call adjMat[0,0]. 
            #Now the matrices use the python indexing
            group[fname[:-len(kind)]]=np.arctanh(adjMat) #normally -14
            #Fisher transform #change to -17 when working with thresholded matrices!!!!
            #Change to -26 for Nonthresholded 
        smooth1[smooth]=group
    return smooth1

def get_all_links(Smoothing,smooth1,length):
    """
    Organizes adjacency matrices in lists by links (distributions of links for the group)

    Parameters
    ----------
    Smoothing : list 
        of Smoothing levels, usually the same folder names used to store the smoothing levels
    smooth : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and subject as sencond keys. The actual matlab matrices/structs etc. as values

    Returns
    -------
    linkSmooth1 : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and subject as sencond keys. The actual matlab matrices/structs etc. as values
    subKeys1 : list
        list of subjects in the order of how the links are stored in the list

    """
    #Verified process :)
    links=[]
    linkSmooth1={key: {} for key in Smoothing}
    x=range(length)
    y=range(length)

    for idx,smooth in enumerate(Smoothing):

        subKeys1=smooth1[smooth].keys()
        for i in x:
            for j in y:
                for subject in subKeys1:
                    links.append(smooth1[smooth][subject][i][j])
                linkSmooth1[smooth]['link_%s_%s' %(i,j)]=links
                links=[]
    return linkSmooth1,subKeys1

def get_regressionMat(filepath,subKeys):
    """
    Read the regression matrices (.csv format) and order the subjects according 
    to the keys order used to order the links in python (get_all_links)
    The .csv file should not have column names.

    Parameters
    ----------
    filepath : list 
        path of the csv file
    subKeys : list
        list of subjects in the order of how the links are stored in the list (get_all_links)

    Returns
    -------
    orgReg : array
        array of the matrix according to subKeys
    """
    #Verified process :)
    #Read the regression matrices
    regress=[]  
    site1=[]
    site2=[]
    site3=[]
    site4=[]
    site5=[]
    site6=[]
    site7=[]
    site8=[]
    fd=[]  
    with open(filepath, 'rt') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            regress.append(row[0])
            site1.append(float(row[1]))
            site2.append(float(row[2]))
            site3.append(float(row[3]))
            site4.append(float(row[4]))
            site5.append(float(row[5]))
            site6.append(float(row[6]))
            site7.append(float(row[7]))
            site8.append(float(row[8]))
            fd.append(float(row[9]))
    regress2=[site1,site2,site3,site4,site5,site6,site7,site8,fd]
    regress2=np.transpose(np.asarray(regress2)) 
    #Organize according to keys
    orgReg=[]
    for subject in subKeys:
        Idx=regress.index(subject)
        orgReg.append(regress2[Idx])
    orgReg=np.asarray(orgReg)
    
    return orgReg

def regress_link(Smoothing,linkSmooth1,linkSmooth2,regMat1,regMat2):
    """
    Regress the site and FD effect from the data (links). The regression matrix 
    X has already been ordered by the function get_regressionMat. 

    Parameters
    ----------
    Smoothing : list 
        of Smoothing levels, usually the same folder names used to store the smoothing levels    
    linkSmooth1, linkSmooth2 : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and links as sencond keys. The list of links as values
    regMat1,regMat2: array
        array of the regression matrices

    Returns
    -------
    link_dist1,link_dist2 : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and links as sencond keys. The regressed links lists as values 
    """
    regressed_links={key: {} for key in Smoothing}
    link_dist1={key: {} for key in Smoothing}
    link_dist2={key: {} for key in Smoothing}
    
    
    
    for idx,smooth in enumerate(Smoothing):
        gKeys=linkSmooth1[smooth].keys() 
        for key in gKeys:
            y=linkSmooth1[smooth][key]+linkSmooth2[smooth][key]
            x=np.concatenate((regMat1,regMat2),axis=0)
            clf = linear_model.LinearRegression(fit_intercept=True)
            clf.fit(x,y)
            betas=clf.coef_
            regressed_links[smooth][key]=y-np.dot(x,betas)-np.ones(len(y))*clf.intercept_
    
    
    for idx,smooth in enumerate(Smoothing):
        gKeys=regressed_links[smooth].keys()
        for key in gKeys:
            size=len(regressed_links[smooth][key])
            
            link_dist1[smooth][key]=regressed_links[smooth][key][0:(size//2+1)]
            link_dist2[smooth][key]=regressed_links[smooth][key][(size//2+1):size]
    
    
    return link_dist1,link_dist2

def links_to_Mat(Smoothing,link_dist,length,subKeys):
    """
    Organize a list of links in Matrix form

    Parameters
    ----------
    Smoothing : list 
        a list of Smoothing levels, usually the same folder names used to store the smoothing levels    
    link_dist : dict
        a dictionary of dictionaries with the Smoothing level as keys, 
        and links as sencond keys. The values of link weights as values 
    length : 
        number of ROIs
    subKeys : list
        list of subjects in the order of how the links are stored in the list (get_all_links)

    Returns
    -------
    regSmooth : dict
        a dictionary of arrays with the Smoothing level as first keys, and subjects as second keys. 
        Regressed matrices are the values.
    """
    #Verified process :)
    
    regSmooth={key: {} for key in Smoothing}
    #subjects=smooth1[Smoothing[1]].keys()
    for idx,smooth in enumerate(Smoothing):
        linkKeys=link_dist[smooth].keys()
        regSmooth[smooth]={key: {} for key in subKeys} #changes order in keys with respect to subKeys
        for idx,subject in enumerate(subKeys):
            regSmooth[smooth][subject]=np.zeros([length,length])
            for link in linkKeys:
                i=int(link.split("_")[1])
                j=int(link.split("_")[2])
                regSmooth[smooth][subject][i][j]=link_dist[smooth][link][idx]
                
        #inds = np.triu_indices_from(regSmooth[smooth],k=1)
        #regSmooth[smooth][(inds[1], inds[0])] = regSmooth[smooth][inds]
        #np.fill_diagonal(regSmooth[smooth], 0)

    return regSmooth
    
if __name__ == "__main__":
    #state all folders and configurations needed
    folder_g1='asd_group/'#ASD
    folder_g2='control_group/'#TC
    filepath_g1='subjects_info_ABIDE_ASD_regress.csv' #Filepath for the regression matrix file 
    #This regression matrix file should contain in order the following columns: subject_ID (str), belong to site I (bool)
    #belong to site II (bool), belong to site III (bool), belong to site IV (bool), mean framewise displacement.
    filepath_g2='subjects_info_ABIDE_controls_regress.csv'
    suffix='-Adj_NoThr_Brainnetome_reg' #name of the suffix of the files to write
    kind='_Adj_NoThr_Brainnetome_check.mat' # '_NoThr.mat', '_NoThr_reg.mat', or '_sphere.mat' suffix of the files to read
    
    #Smoothing=['4','6','8','10','12','14','16','18','20','22','24','26','28','30','32']
    Smoothing=['0']
    Smoothing=['Brainnetome_'+s+'mm/' for s in Smoothing]
    length=246 #number of ROIs
    
    #Run the computations for both groups
    smooth1=read_all_subjects(Smoothing, folder_g1,kind) #Read all subjetcs for a group:ASD  
    linkSmooth1,subKeys1=get_all_links(Smoothing,smooth1,length) #Organize info by links
    regMat1=get_regressionMat(filepath_g1,subKeys1)#Order the regression matrix according to how python is reading the files
    
    smooth2=read_all_subjects(Smoothing, folder_g2,kind)    
    linkSmooth2,subKeys2=get_all_links(Smoothing,smooth2,length)  
    regMat2=get_regressionMat(filepath_g2,subKeys2)
    
    #Regress all the links for all the smoothing levels
    link_dist1,link_dist2=regress_link(Smoothing,linkSmooth1,linkSmooth2,regMat1,regMat2)
    
    
    regSmooth1=links_to_Mat(Smoothing,link_dist1,length,subKeys1)
    regSmooth2=links_to_Mat(Smoothing,link_dist2,length,subKeys2)
    
    #save regressed matrices
    for smooth in Smoothing:
        subjects=regSmooth1[smooth].keys()
        for idx,subject in enumerate(subjects):
            scipy.io.savemat(folder_g1+smooth+'/'+subject+suffix,{'Adj':regSmooth1[smooth][subject]})
    
    for smooth in Smoothing:
        subjects=regSmooth2[smooth].keys()
        for idx,subject in enumerate(subjects):
            scipy.io.savemat(folder_g2+smooth+'/'+subject+suffix,{'Adj':regSmooth2[smooth][subject]})
