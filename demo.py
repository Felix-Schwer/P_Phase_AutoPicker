# -*- coding: utf-8 -*-
"""
Demo script for P_Phase_AutoPicker by Felix Schwer (2023)

Plots in this script are supposed to be displayed inline
Alternatively the output of the visualize functions can be saved through their return values.

The test example data is taken from https://examples.obspy.org/
"""

import pphaseAutoPicker as ppap
from pathlib import Path

reader = ppap.File_Reader('.')  # Reader instace to manage file paths containing seismological data
                                # and read various data formats with obspy
                                
print(reader.PathSet)           # All the paths the Reader will currently try to read data from

reader.exclude_paths('.')       # Manage the set of these paths with the include/exclude_paths methods
reader.include_paths( [Path.cwd() / 'trigger_data' / '000000000_0036EE80.mseed', # The reader instance can reed from files aswell
                       Path.cwd() / 'trigger_data' / 'data'] )                   # as from directories containing seismological data files.
                                                                                 # Arguments can be string representations of paths, 
                                                                                 # pathlib.Path objects or os.path representations either as a 
                                                                                 # single argument or contained in an iterable
# With extract_timeseries_data() the reader instance will try to read data
# from all provided file paths with the obspy package and return all the
# relevant data for the Picker class to operate with                                                                                 
trace_datasets, sampling_intervals, header_datasets, _ = reader.extract_timeseries_data()

# Instantiate a new Picker instance using one extracted dataset from the example files
p = ppap.Picker(trace_datasets[0],sampling_intervals[0], header_datasets[0])

p.visualize_pick_info('Example trace data')      # Visualize only the trace data

p.pick_routine_Allen()                                      # Run both of the available pick routines
p.pick_routine_BK()
p.visualize_pick_info('Example picks')           # Visualize the trace together with the picks
                                                            
# It may be the case that the algorithms did not find a pick, even though you'd expect them to
# To find out more about that, run the picking routines again with the parameter return_insights = True
pick,t = p.pick_routine_Allen(return_insights=True)
p.pick_routine_BK(return_insights=True)
p.visualize_pick_info()

# To get better results, you can adjust the parameters used by the picker.
# There are different presets of parameters available taken from Allen (1978),
# Baer and Kradolfer (1987) and Küperkoch et. al. (2012) for different categories
# of events. Use the keywords 'local', 'regional' or 'tele'

# Showcase this on a new example
q = ppap.Picker(trace_datasets[42],sampling_intervals[42], header_datasets[42])
q.pick_routine_Allen(return_insights=True)
q.visualize_pick_info('Default paramters')
q.pick_routine_Allen(return_insights=True,parameter_preset='tele')
q.visualize_pick_info('Preset teleseismic parameters')

# These presets are adjustable via the following class method.
# They will be changed accross all Picker instances (see when the routine is run on p)
ppap.Picker.modify_Parameter_Presets('local',S1=5.,TUp=0.5,TDown=0.2)
# Call with no keyword arguments to simply read the current parameters for a category
ppap.Picker.modify_Parameter_Presets('regional')

# The pick routines can also be called explicitely specifiying the parameters to be used
# This will override any of the other preset values
p.pick_routine_Allen()
p.pick_routine_BK(S1=2.,parameter_preset='local',return_insights=True)
fig, ax = p.visualize_pick_info('New parameters')   # Compare the parameters of the BK picker to the parameters on
                                                    # the first example run of the routine
                                                    




"""
References
----------
    Rex V. Allen (1978): Automatic earthquake recognition and timing from single traces.
    Bulletin of the Seismological Society of America; 68 (5): 1521–1532.
    https://doi.org/10.1785/BSSA0680051521
    
    M. Baer, U. Kradolfer (1987): An automatic phase picker for local and teleseismic events.
    Bulletin of the Seismological Society of America 1987; 77 (4): 1437–1445.
    https://doi.org/10.1785/BSSA0770041437
    
    Küperkoch, L., Meier, T., Diehl, T. (2012): Automated Event and Phase Identification.
    In: Bormann, P. (Ed.), New Manual of Seismological Observatory Practice 2 (NMSOP-2),
    Potsdam : Deutsches GeoForschungsZentrum GFZ.
    https://doi.org/10.2312/GFZ.NMSOP-2_ch16
"""                
                            
