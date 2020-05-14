import os
import sys
import warnings
import numpy as np
import pandas as pd
from glob import glob
from astropy.io import ascii
from ps1_tools import ps1cone, angle_sep

def query(ra_center,dec_center,radius):
    '''
    Inputs
    ------
    ra_center: float [degrees]
    dec_center: float [degrees]
    radius: float [degrees]

    '''
# Create an Empty DataFrame
    columns = """objID,raMean,decMean,nDetections,ng,nr,ni,nz,ny,
            gMeanPSFMag,gMeanPSFMagErr,rMeanPSFMag,rMeanPSFMagErr,
            iMeanPSFMag,iMeanPSFMagErr,zMeanPSFMag,zMeanPSFMagErr,
            yMeanPSFMag,yMeanPSFMagErr,gMeanKronMag,gMeanKronMagErr,
            rMeanKronMag,rMeanKronMagErr,iMeanKronMag,iMeanKronMagErr,
            zMeanKronMag,zMeanKronMagErr,yMeanKronMag,yMeanKronMagErr""".split(',')
    columns = [x.strip() for x in columns]
    columns = [x for x in columns if x and not x.startswith('#')]

# Query Constraints
    qpath = './'
    constraints = {'ng.gt':4,  # Minimum 5 Detections per filter
                   'nr.gt':4,
                   'ni.gt':4,
                   'nz.gt':4,
                   'ny.gt':4}

# Perform the Cone Search
    results = ps1cone(ra_center,dec_center,radius,release='dr2',columns=columns,**constraints)

# Convert Results into an Astropy Table, improve formatting,
# and then create a Pandas Table
    apy_tab = ascii.read(results)
    for f in 'grizy':
        col1 = f+'MeanPSFMag'
        col2 = f+'MeanPSFMagErr'
        try:
            apy_tab[col1].format = ".4f"
            apy_tab[col1][apy_tab[col1] == -999.0] = np.nan
        except KeyError:
            print("{} not found".format(col1))
        try:
            apy_tab[col2].format = ".4f"
            apy_tab[col2][apy_tab[col2] == -999.0] = np.nan
        except KeyError:
            print("{} not found".format(col2))
    ps_tab = apy_tab.to_pandas()

# Save the query for future use
    ps_tab.to_csv(qpath+'ps_query.csv',index=False)
    
    return ps_tab
