"""
P_Phase_AutoPicker
------------------
    A python module that implements basic seismic P-Phase picking algorithms, aimed at
    automatically detecting seismic events from seismogramm data and estimating P-Phase arrival times.
    Created by Felix Schwer (2023)

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
    
Classes
-------
    File_Reader: Manage paths of seismological data and read data files with obspy
    Picker: Routines to detect seismic events and P-Phase arrival times and visualize them
"""

import obspy as op                              # Read and Process seismological data from various data formats
import numpy as np                              # Mathematical operations and calculations, use of Numpy arrays
from pathlib import Path                        # Handle Filepaths to read data
import matplotlib as mpl                        # Visualize data and pick routines
import matplotlib.pyplot as plt                 # "
from matplotlib.ticker import MultipleLocator   # "
import drawpicker as dp                         # Seperate Module containing helper functions to plot data


class File_Reader:
    """
    A class to manage differnt paths containing seismological data and to read this data with obspy

    Attributes
    ----------
        PathSet: set{pathlib.Path}
            A set containing all the paths as pathlib.Path objects the user is expecting to read seismological data from

    Public Methods
    --------------
        include_paths(path)
        exclude_paths(path)
        extrac_timeseries_data() -> trace_datasets,sampling_intervals,header_datasets,trace_objects
    """
    
    def __init__(self, path):
        """
        Parameters
        ----------
        path : A singular string, pathlib.Path object or os.PathLike object OR an iterable (list/set/tuple) containing multiple of them
            Path of a seismological data file or a folder containing such data files

        Returns
        -------
        None.
        
        """
        self.PathSet = set()                    # Initialize empty set
        if isinstance(path, (list,set,tuple)):
            paths = {Path(p) for p in path}     # Convert contents of iterable to Path objects
            self.PathSet.update(paths)          # and add to set
        else:
            self.PathSet.add(Path(path))        # Add singular path object to set
    
    def include_paths(self, path):
        """
        Adds more paths to the PathSet variable to later read data from

        Parameters
        ----------
        path : A singular string, pathlib.Path object or os.PathLike object OR an iterable (list/set/tuple) containing multiple of them
            Path(s) of one or more seismological data files or a folder containing such data files

        Returns
        -------
        None.

        """
        if isinstance(path, (list,set,tuple)):
            paths = {Path(p) for p in path}     # Convert contents of iterable to Path objects
            self.PathSet.update(paths)          # and add to set
        else:
            self.PathSet.add(Path(path))        # Add singular path object to set

    def exclude_paths(self, path):
        """
        Removes paths from the PathSet variable, so this data will no longer be read 

        Parameters
        ----------
        path : A singular string, pathlib.Path object or os.PathLike object OR an iterable (list/set/tuple) containing multiple of them
            Path(s) of one or more seismological data files or a folder containing such data files

        Returns
        -------
        None.

        """
        if isinstance(path, (list,set,tuple)):
            paths = {Path(p) for p in path}         # Convert contents of iterable to Path objects
            self.PathSet.difference_update(paths)   # and remove from set
        else:            
            self.PathSet.discard(Path(path))        # Remove singular path object from set
        
    def extract_timeseries_data(self):
        """
        Reads seismological data from the locations given by the PathSet variable and returns relevant data
        for the Picker class to operate with (Will not read from subfolders of a given directory!)

        Raises
        ------
        ValueError
            When obspy fails to read data from one of the provided path locations.

        Returns
        -------
        trace_datasets : [numpy.ndarray]
            A list of all the seismological traces, which were read.
        sampling_intervals : [float]
            Corresponding sampling intervals for each trace in seconds [s].
        header_datasets : [dict]
            Corresponding metadata for each trace in form of an obspy generated header dictonary.
        trace_objects : [obspy.core.trace.Trace]
            A list of the obspy Trace objects corresponding to each trace. This is useful if
            the user wishes to do some preprocessing (e.g. filter, ...) on the trace with the
            methods provided by the obspy Trace class. The trace data, sampling intervall and
            header data will then have to be extraced manually from the Trace object to pass
            it to the Picker class
        """
        
        Streams = [] # Append all of the op.core.stream.Stream instances that op.read returns
        
        for path in self.PathSet: # Read from all the Paths currently in PathSet variable
            
            # Attempt to read from file
            if path.is_file():
                try:
                    st = self.generic_file_reading_routine(path) # returns op.core.stream.Stream object from filepath
                except ValueError:
                    
                    #For internal Dev: Add a more specific reading routine here if op.read() (from op.core.stream) struggles with gse2-Format.
                    #op.core.stream.read() uses op.io.gse2.core._read_gse2() for gse2 but somehow, sometimes
                    #this doesn't work, but instead explicitly using op.io.gse2.libgse2.read() does.
                    #op.io.gse2.libgse2.read() does not return op.core.stream.Stream objects, so the returned object
                    #can't be appended to 'Streams' and the data has to be extracted from the object seperately
                    raise #Reraise any caught errors here, as long as the above is not implemented
                
                else: # Attempt to read from file was succesful
                    Streams.append(st)
            
            # Read from Directory
            elif path.is_dir():
                dir_contentsList = sorted(path.glob('*')) # Get all contents of directory
                files = [f for f in dir_contentsList if f.is_file()] # Only read from files and ignore all subfolders
                
                # Now attempt to read from all files of the directory
                for filepath in files:
                    try:
                        st = self.generic_file_reading_routine(filepath) # returns op.core.stream.Stream object from filepath
                    except ValueError:
                        
                        #For internal Dev: Add a more specific reading routine here if op.read() (from op.core.stream) struggles with gse2-Format.
                        #op.core.stream.read() uses op.io.gse2.core._read_gse2() for gse2 but somehow, sometimes
                        #this doesn't work, but instead explicitly using op.io.gse2.libgse2.read() does.
                        #op.io.gse2.libgse2.read() does not return op.core.stream.Stream objects, so the returned object
                        #wouldn't be able to be appended to 'Streams' and the data would have to be extracted from the object seperately
                        raise #Reraise any caught errors here, as long as the above is not implemented
                    
                    else: # Attempt to read from file was succesful
                        Streams.append(st)
        
        # Extract all the relevant data for the Picker class from the Stream Objects (see docstrings)
        trace_datasets = []
        sampling_intervals = []
        header_datasets = []
        trace_objects = []
        for st in Streams:
            for tr in st:
                # For each trace in each stream append this data
                trace_objects.append(tr)
                trace_datasets.append(tr.data)
                sampling_intervals.append(tr.stats.delta)
                header_datasets.append(tr.stats)
                
        return trace_datasets,sampling_intervals,header_datasets,trace_objects

    @staticmethod
    def generic_file_reading_routine(filepath):
        """
        Helper function to return a obspy Stream object from reading a seismolical data file
        Uses obspy.core.stream.read()

        Parameters
        ----------
        filepath : pathlib.Path object
            Has to be a file containing seismological data in a dataformat supported by obspy.
            https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
            (Supported with obspy 1.4.0: AH, ALSEP_PSE, ALSEP_WTH, ALSEP_WTN, CSS, DMX, GCF,
            GSE1, GSE2, KINEMETRICS_EVT, KNET, MSEED, NNSA_KB_CORE, PDAS, PICKLE, Q, REFTEK130,
            RG16, SAC, SACXY, SEG2, SEGY, SEISAN, SH_ASC, SLIST, SU, TSPAIR, WAV, WIN, Y)
            
        Returns
        -------
        st : obspy.core.stream.Stream object

        """
        try: #Simply try to read the file
            st = op.read(filepath)
        except (ValueError, TypeError):
            try: #If this doesn't work, try to guess the file format from the path suffix and try again
                fmt = filepath.suffix.replace('.', '')
                st = op.read(filepath, format=fmt)
            except Exception:
                raise
            else: #op.core.stream.read() was succesful, so the returned object should be an op.core.stream.Stream instance
                assert isinstance(st, op.core.stream.Stream)
                return st
        else: #op.core.stream.read() was succesful, so the returned object should be an op.core.stream.Stream instance
            assert isinstance(st, op.core.stream.Stream)
            return st


            
class Picker:
    """
    A class to run automated P-Phase Picking routines on seismic trace data to detect
    events and estimate P-Phase arrival times and to visualize trace and pick data
    
    Attributes
    ----------
        trace: numpy.ndarray(dtype=float)
            Stores the trace data
        dt: float
            Sampling intervall in seconds [s]
        header: dict
            Contains an obspy generated header dictonary
        pick_Allen: int or None
            Is only set after running the Allen pick routine. Stores the index of the
            first succesfully accepted pick flag of the algorithm. If None, no pick
            flag was usccesfully accepted.
        insights_Allen: tuple
            Is only set after running the Allen pick routine with parameter
            return_insights = True. Contains a tuple of the form
            (CF, E, sta, lta, delta, M, L, s, params)
            containing all relevant data that was computed during the routine.
            params is a tuple of the form
            (alpha,beta,gamma,tMin,MMin)
            containg all input parameters that were used to run the routine
        pick_BK: int or None
            Is only set after running the BK pick routine. Stores the index of the
            first succesfully accepted pick flag of the algorithm. If None, no pick
            flag was usccesfully accepted.
        insights_BK: tuple
            Is only set after running the BK pick routine with parameter
            return_insights = True. Contains a tuple of the form
            (CF, E, cumultative_mean, dynamic_variance, params)
            containing all relevant data that was computed during the routine.
            params is a tuple of the form
            (TUp,TDown,S1)
            containg all input parameters that were used to run the routine
        time_axis: numpy.ndarray
            Is only set after running the 'visualize' routine
            Stores the time in seconds [s] aftet the beginning of the trace
            for each datapoint
    
        Parameter_Presets: dict
            Class variable containing multiple parameter presets for different types
            of event categories ('local,'regional' and 'tele' for teleseismic) in form
            of a nested dictonary. The picking routines can then use these parameters
            by default if they have such an event type assigned to them. Also contains
            default values if no event category is specified and the picking routine
            is called without parameters.
            This is accessed via the Class not the Picker instances. Changes to the
            parameter presets therefore apply to all Picker instances!
            
    Public Methods
    --------------
        pick_routine_Allen(alpha=None,beta=None,gamma=None,tMin=None,MMin=None
                           parameter_preset=None,weighted_average='exp',return_insights=False) -> pick_flag (, insights_Allen)
        pick_routine_BK(TUp=None,TDown=None,fCornerLow=None,fCornerHigh=None,
                        S1=None,parameter_preset=None,CF_denominator='std',return_insights=False) -> pick_flag (, insights_BK)
        visualize_pick_info(event_name='') -> fig,ax
        
        @classmethod
        modify_parameter_presets(category,**kwargs)
    
    """
    def __init__(self,trace,sampling_interval,header):
        """
        Constructs a Picker object using the data of one trace in the required
        form described below. After this the picking routines can be run from the instance

        Parameters
        ----------
        trace : numpy.ndarray
            Numpy array containing the trace data
        sampling_interval : float
            Sampling interval of the trace in seconds [s]
            Regular sampling required!
        header : dict
            obspy genereated header dictonary

        Returns
        -------
        None.

        """
        self.trace = trace.astype(float)
        self.dt = sampling_interval
        self.header = header
    
    # Parameter Preset definition for all instances of the picker. Access to change all
    # presets is given via the public modify_Parameter_Presets method
    Parameter_Presets = {'default':{'alpha': 0.05,         # If nothing else is specified
                                     'beta': 5.,
                                     'gamma': 5.,
                                     'tMin': 1.5,
                                     'MMin': 40,
                                     'fCornerLow': 2/3,
                                     'fCornerHigh': 20,
                                     'TUp': 1.5,
                                     'TDown': 0.35,
                                     'S1': 10.},
                         'local':{'alpha': 0.1,             # Local events. Allen parameter from (Küperkoch et. al. 2012)
                                  'beta': 5.,
                                  'gamma': 2.,
                                  'tMin': 3.,
                                  'MMin': 40,
                                  'fCornerLow': 1.,         # Corner frequences adapted from (Küperkoch et. al. 2012)
                                  'fCornerHigh': 30.,
                                  'TUp': 1.,                # Event duration times calculated from corner frequencies after
                                  'TDown': 0.25,            # (Baer & Kradolfer 1987)
                                  'S1': 10.},
                         'regional':{'alpha': 0.1,          # Regional seismic events, same references as above
                                     'beta': 5.,
                                     'gamma': 2.,
                                     'tMin': 3.,
                                     'MMin': 40,
                                     'fCornerLow': 0.5,
                                     'fCornerHigh': 15.,
                                     'TUp': 2.,
                                     'TDown': 0.5,
                                     'S1': 10.},
                         'tele':{'alpha': 10.,              # Teleseismic events, same references as above
                                 'beta': 50.,
                                 'gamma': 3.,
                                 'tMin': 40.,
                                 'MMin': 40,
                                 'fCornerLow': 0.1,
                                 'fCornerHigh': 2.,
                                 'TUp': 10.,
                                 'TDown': 2.5,
                                 'S1': 10.}}
    
    @classmethod
    def modify_Parameter_Presets(cls, category, **kwargs):
        """
        Interface that allows the user to change the preset parameters used in the
        picking routines across all Picker instances. If called from an instance
        as opposed to the Picker class a shadow variable for this instance will be created.
        This is not intended behaviour!
        Call with no keyword arguments to simply read the current parameters for a category

        Parameters
        ----------
        category : str
            Either 'default', 'local', 'regional' or 'tele' depending on which category should be modified
        **kwargs : keyword arguments of the form parameter = [float]
            Parameters of the given category that should be changed and the value they
            should be changed to. Possible keywords (parameters) are:
                alpha, beta, gamma, tMin, MMin, fCornerLow, fCornerHigh, TUp, TDown, S1
            Call with no keyword argument so simply read out the current values
                
            If the corner frequencies are specified, the event duration times TUp, TDown for the BK Picker
            are calculated after (Baer & Kradolfer 1987) as
                TUp = 1/fCornerLow, TDown = (1/fCornerLow + 1/fCornerHigh)/4
            In this case both freqeuncies need to be specified and neither TUp nor TDown can be given as a paramter!
            (Using corner frequencies may be more intuitive for the user, if a signal in a certain frequency range
             is expected or if the signal has been band pass filtered)

        Returns
        -------
        None.

        """
        if (('fCornerLow' in kwargs) or ('fCornerHigh' in kwargs)) and (('TUp' in kwargs) or ('TDown' in kwargs)):
            raise ValueError('Either only corner frequencies or event durations can be modified at the same time')
        
        # Change parameters in 'category' dictonary
        for parameter in kwargs:
            cls.Parameter_Presets[category][parameter] = kwargs[parameter]
        
        # If corner frequencies have been given: Calculate event duration times of BK Picker according to (Baer & Kradolfer 1987)
        # Both frequencies need to be given as a keyword argument to avoid miscalculations!
        if ('fCornerLow' in kwargs) or ('fCornerHigh' in kwargs):
            cls.Parameter_Presets[category]['TUp'] = 1/kwargs['fCornerLow']
            cls.Parameter_Presets[category]['TDown'] = ((1/kwargs['fCornerLow'])+(1/kwargs['fCornerHigh']))/4
        # If event duration times are given: Automatically adjust to the corresponding frequencies too.
        # (They don't necessarily play a role for the picker, but this is done to mantain a coherent set of parameters)
        elif ('TUp' in kwargs) or ('TDown' in kwargs):
            cls.Parameter_Presets[category]['fCornerLow'] = 1/cls.Parameter_Presets[category]['TUp']
            try:
                cls.Parameter_Presets[category]['fCornerHigh'] = 1/(4*cls.Parameter_Presets[category]['TDown'] - cls.Parameter_Presets[category]['TUp'])
            except ZeroDivisionError:
                cls.Parameter_Presets[category]['fCornerHigh'] = np.inf
            
        print(f'Parameter Preset for {category} is set to:\n{cls.Parameter_Presets[category]}\nfor all instances')
    
    def calc_event_times_from_corner_freq(self,TUp,TDown,fCornerHigh,fCornerLow):
        """
        Helper function to sort out the arguments given to the pick_routine_BK
        Calculates event duration times according to (Baer & Kradolfer 1987) if
        corner freqencies are given as parameters.

        Parameters
        ----------
        TUp : int or float
        TDown : int or float
        fCornerHigh : int or float > 0
        fCornerLow : int or float > 0, < fCorner High
            
        Raises
        ------
        ValueError
            If event duration times and corner frequencies are both given at the same time
            or if not both corner frequencies are specified

        Returns
        -------
        TUp : int or float
        TDown : int or float
            Event duration time parameters in s to be used in the BK pick routine.

        """
        if ((TUp is not None) or (TDown is not None)) and ((fCornerHigh is not None) or (fCornerLow is not None)):
            raise ValueError('Either only corner frequencies or event durations can be given as arguments')
        elif ((fCornerHigh is not None) and (fCornerLow is None)) or ((fCornerHigh is None) and (fCornerLow is not None)):
            raise ValueError('To calculate the event duration parameters both corner frequencies need to be specified')
        if (fCornerHigh is not None) and (fCornerLow is not None):
            # Calculate event duration times of BK Picker based of Corner frequencies according to (Baer & Kradolfer 1987)
            assert (fCornerHigh > fCornerLow) and ((fCornerHigh > 0) and (fCornerLow > 0))
            TUp = 1/fCornerLow
            TDown = ((1/fCornerLow)+(1/fCornerHigh))/4
        else:
            TUp = TUp
            TDown = TDown
        return TUp,TDown
    
    def apply_preset_values(self,preset,**kwargs):
        """
        Helper function to sort out whether any parameters have been given to the pick routine
        itself or whether any of the parameter presets should be used (default, if no event
        category is specified)

        Parameters
        ----------
        preset : str or None
            Keyword argument containing the argument passed to paramter_preset of the pick routine
        **kwargs : Other keyword arguments passed to the pick routine

        Raises
        ------
        ValueError
            If parameter_preset does not equal either 'local', 'regional', 'tele' or None

        Returns
        -------
        tuple
            tuple of the parameters that the pick routine can now use to run

        """
        for kw in kwargs:
            if kwargs[kw] is None:
                if preset is None:
                    kwargs[kw] = Picker.Parameter_Presets['default'][kw]
                elif preset == 'local' or preset == 'regional' or preset == 'tele':
                    kwargs[kw] = Picker.Parameter_Presets[preset][kw]
                else:
                    raise ValueError("'Parameter Preset name not definied. Choose out of preset = 'local', 'regional' or 'tele'")
        return tuple(kwargs.values())
    
    #-------------------------------------------------------------------------#
    #                   PICKING ALGORITHMS BEGIN HERE                         #
    #-------------------------------------------------------------------------#
    def pick_routine_Allen(self,alpha = None,beta = None,gamma = None,tMin = None,MMin = None,parameter_preset=None,weighted_average='exp',return_insights=False):
        """
        Algorithm to run the automatic P-Phase Picker of (Allen 1978).
        See (Allen 1978) and (Küperkoch et. al 2012) for more detailed descriptions.

        Parameters
        ----------
        alpha : int or float, optional
            Time in seconds [s] to use for the short term average calculation. If set, this will override any preset values
        beta : int or float, optional
            Time in seconds [s] to use for the long term average calculation. If set, this will override any preset values
        gamma : int or float, optional
            Threshold for the CF to reach, to trigger a pick flag. If set, this will override any preset values
        tMin : inf or float, optional
            Minimum event duration in seconds [s] to accept a pick flag as a genuine event.
            If set, this will override any preset value
        MMin : int, optional
            Minimum amount of trace zero crossings during the event to accept a pick flag as a genuine event.
            If set, this will override any preset value
        parameter_preset : str, optional
            Use the preset parameters if they are not specified otherwise. Can be either 'local', 'regional' or 'tele'.
            If not set, 'default' will be used
        weighted_average : str, optional
            To calcualte the short- and longterm averages use either a uniformly weighted rolling average ('uni')
            or an exponentially weighted moving average ('exp') as in (Allen 1978). The exponential weighting constant
            is calculated from alpha in a way, so that it approximates the rolling average of window lenght alpha.
            The default is 'exp'.
        return_insights : bool, optional
            If set to True the function will return a tuple containing all relevant data calculated during
            the algorithm and save this tuple to an instance variable too. The default is False.

        Returns
        -------
        ret : int or None (, tuple)
        -   If return_insights is False, then only the pick flag of the first succesfully accepted pick will
            be returned. If no pick was accepted, None is returned.
            The pick flag represents an index number of the numpy.ndarray storing the trace data, not a time!
        -   If return_insights is True, (pick_flag, insights_Allen) will be returned, where insights_Allen is
            a tuple of the form (CF, E, sta, lta, delta, M, L, s, params) containing all relevant data that
            was computed during the routine. params in itself is a tuple of the form (alpha,beta,gamma,tMin,MMin)
            of the parameters passed in the function call.
                The variable names were chosen to follow (Allen 1978) and (Küperkoch et. al. 2012) as closely as possible.
                I deviate in the naming of the characterstic function (CF), under which I understand the function, that
                is able to trigger a pick flag, when exceeding a certain threshold in accordance with (Baer & Kradolfer 1987).
                (Allen 1978) names his sqaured envelope estimation, the CF.
                Refer to (Allen 1978) and (Küperkoch et. al. 2012) and the comments in the code for the meaning of these outputs.
        """
        
        # Sort out the parameters (Apply preset or default values if not specified otherwise) and save them
        alpha,beta,gamma,tMin,MMin = self.apply_preset_values(preset=parameter_preset,alpha=alpha,beta=beta,gamma=gamma,tMin=tMin,MMin=MMin)
        params = alpha,beta,gamma,tMin,MMin
        
        # Compute characteristic function (CF)
        Duration_to_accept_pick = int(np.ceil(tMin/self.dt))        # number of indices after which an event can be accepted as a succesful pick
        
        E = self.calc_Allen_envelope()                              # e**2, where e is an estimation of the envelope of the trace.
                                                                    # (Allen 1978) uses a fixed weighing constant for this, but I am following
                                                                    # the approach of (Küperkoch et. al. 2012) (see docstrings of calc_Allen_envelope)
                                                                    
        sta = self.STA(E,self.dt,alpha,weights=weighted_average)    # Compute short and long term averages of the function e**2
        lta = self.LTA(E,self.dt,beta,weights=weighted_average)
        
        CF = np.nan_to_num(sta/lta,nan=0.,posinf=0.,neginf=0.)      # The charateristic Function CF is defined as the quotient of the short term
                                                                    # and long term averages over the squared envelope. (Allen 1978) uses an
                                                                    # exponentially weighted average, which is set as the default here (see docstrings of STA/LTA)
        
        # Initialize event evaluation variables
        pick_flag = None
        M = np.zeros_like(self.trace,dtype=int)                     # Number of trace zero crossings during the estimated event duration. This
                                                                    # serves as an estimation for the number of trace peaks during an event.
                                                                    # It is used to compute the Termination number L
        
        L = 3*np.ones_like(self.trace,dtype=int)                    # Termination number. Is computed as L=3+M/3 after (Allen 1978). This is
                                                                    # used to judge when an event should be defined as 'over'. For an event to
                                                                    # be over, the short term average has to be below the continuation criterion
                                                                    # delta for L times in a row. L increases as the event goes on (and M increases).
                                                                    # This is done to be able to reject short noise increases rather quickly as L
                                                                    # will be a small number at the beginning.
                                                                    
        s = np.zeros_like(self.trace,dtype=int)                     # s counts the number of succesive trace zero crossings in a row, where the short
                                                                    # average does not exceed the continuation criterion delta
                                                                    
        delta = np.zeros_like(self.trace,dtype=float)               # Continuation criterion: If the short term average falls below delta for a continued
                                                                    # amount of time, this is used as an indication, that the event should be jugded as
                                                                    # 'over'. I compute delta after (Küperkoch et. al. 2012) as a function of lta[pick_flag], 
                                                                    # the long term average when the pick flag was triggered. This serves as an estimation
                                                                    # for the noise level before the event. delta linearally increases with M as the event goes on.
        
        # Start to evaluate trace
        for i in range(len(self.trace)):
            M[i] = M[i-1] # The evaluation variables are only updated for each zero crossing during an event. Otherwise the just stay the same
            L[i] = L[i-1]
            s[i] = s[i-1]
            delta[i] = delta[i-1]
            
            # If no pick flag is currently set, check whether the CF has increased above the treshold gamma and a new pick flag should be triggered
            if pick_flag is None: 
                if CF[i] >= gamma:
                    pick_flag = i
            
            # If a pick flag is currently set, evaluate the ongoing event
            else:
                # Evaluate only at the zero crossings of the trace
                if (np.sign(self.trace[i])*np.sign(self.trace[i-1]) <= 0): 
                    
                    M[i] = M[i] + 1                 # Update trace zero crossings number
                    L[i] = 3 + M[i]/3               # Compute new Termination number after (Allen 1978)
                    delta[i] = lta[pick_flag]*M[i]  # Compute new continuation criterion after (Küperkoch et. al. 2012) and as explained above
                    
                    if sta[i] >= delta[i]:          # Evaluate continuation criterion as discussed above (Küperkoch et. al. 2012)
                        s[i] = 0                    # In 's': Count succesive zero crossings, where continuation criterion is not met (Allen 1978)
                    else:
                        s[i] = s[i] + 1
                        
                    if s[i] >= L[i]:                # If 's' reaches the Termination number, the event is declared 'over'.
                        # In this case: Evaluate whether it should be rejected or accepted as a genuine P-Phase pick of a seismic event
                        
                        # To accept, the event has to be longer than the minimum event duration tMin and has to include a minimum amout of 
                        # trace zero crossings MMin (In order to reject short noise increase events).
                        if (i-pick_flag) >= Duration_to_accept_pick and M[i] >= MMin:
                            # Return the pick represeted by its index and possibly return additional info about all of the computed variables.
                            self.pick_Allen = pick_flag
                            if return_insights:
                                self.insights_Allen = (CF, E, sta, lta, delta, M, L, s, params)
                            else: # Delete insights from an old run of the routine, so there is no confusion
                                try:
                                    delattr(self, 'insights_Allen')
                                except AttributeError:
                                    pass
                            ret = (pick_flag, (CF, E, sta, lta, delta, M, L, s, params)) if return_insights else pick_flag
                            return ret
                        
                        # Otherwise reject the pick, reset all evaluation parameters and look for a new trigger in the characteristic function.
                        else:
                            pick_flag = None
                            M[i] = 0
                            L[i] = 3
                            s[i] = 0
                            delta[i] = 0
        
        # If no pick flag was succesfully accepted return None and possibly additional info about all of the computed variables.
        self.pick_Allen = None
        if return_insights:
            self.insights_Allen = (CF, E, sta, lta, delta, M, L, s, params)
        else: # Delete insights from an old run of the routine, so there is no confusion
            try:
                delattr(self, 'insights_Allen')
            except AttributeError:
                pass
        ret = (None, (CF, E, sta, lta, delta, M, L, s, params)) if return_insights else None
        return ret
        
    
    def pick_routine_BK(self,TUp = None,TDown = None,fCornerHigh = None,fCornerLow = None,S1=None,parameter_preset=None,return_insights=False):
        """
        Algorithm to run the automatic P-Phase Picker of (Bear & Kradolfer 1987).
        See (Bear & Kradolfer 1987) and (Küperkoch et. al 2012) for more detailed descriptions.

        Parameters
        ----------
        TUp : int or float, optional
            Time in seconds [s], that the characterstic function has to stay above
            the threshold S1 for a pick flag to be accepted as a genuine P-Phase pick.
            If set, this will override any preset values
        TDown : int or float, optional
            Time in seconds [s], that the characterstic function is cumulatively allow
            to drop below the threshold S1 during TUp without the event being rejected.
            If set, this will override any preset values
        fCornerHigh : int or float, optional, > fCornerLow > 0
            Instead of passing TUp and TDown explicitly you can pass the corner frequencies
            of the expected signal bandwidth and let TUp and TDown be calculated after
            (Baer & Kradolfer 1987) from this. This e.g. applies if the signal has been
            band pass filtered. If set, this will override any preset values.
            High corner frequency in Hertz [Hz].
            Both frequencies need to be specified for this calculation!
        fCornerLow : int of float, optional
            See above. Low corner frequency in Hertz [Hz].
            Both frequencies need to be specified for this calculation!
        S1 : int or float, optional
            Threshhold for the characterstic function to exceed to trigger a pick flag.
            TIf set, this will override any preset values..
        parameter_preset : str, optional
            Use the preset parameters if they are not specified otherwise. Can be either
            'local', 'regional' or 'tele'. If not set, 'default' will be used
       return_insights : bool, optional
            If set to True the function will return a tuple containing all relevant data calculated during
            the algorithm and save this tuple to an instance variable too.. The default is False.

        Returns
        -------
        ret : int or None (, tuple)
        -   If return_insights is False, then only the pick flag of the first succesfully accepted pick will
            be returned. If no pick was accepted, None is returned.
            The pick flag represents an index number of the numpy.ndarray storing the trace data, not a time!
        -   If return_insights is True, (pick_flag, insights_BK) will be returned, where insights_BK is
            a tuple of the form (CF, E, cumultative_mean, dynamic_variance, params) containing all relevant data that
            was computed during the routine. params in itself is a tuple of the form (TUp,TDown,S1) of the parameters passed in the function call.
                The variable names were chosen to follow (Baer & Kradolfer 1987) and (Küperkoch et. al. 2012),
                but to still fit with the variables already established for the Allen Picker
                Refer to (Baer & Kradolfer 1987) and (Küperkoch et. al. 2012) and the comments in the code for the meaning of these outputs.

        """
        
        # Sort out the time and frequency values passed to the function and calculate on from the other if necessary
        TUp,TDown = self.calc_event_times_from_corner_freq(TUp,TDown,fCornerHigh,fCornerLow)
        # Sort out the parameters (Apply preset or default values if not specified otherwise) and save them
        TUp,TDown,S1 = self.apply_preset_values(preset=parameter_preset,TUp=TUp,TDown=TDown,S1=S1)
        params=TUp,TDown,S1
        assert TUp > TDown
        
        # Initialize all variables
        window_length_to_evaluate = int(TUp/self.dt)    # Number of indices an event must stay above the threshold S1 with its CF for the pick to be accepted
        window_length_grace_period = int(TDown/self.dt) # Number of indices it is allowed to fall below the threshold S1 without rejecting the pick
        
        E = np.nan_to_num(self.calc_BK_envelope())      # An estimate at the squared envelope e**2 of the trace, calculated after (Baer & Kradolfer 1987)
        sum_E = np.cumsum(E)                            # Cumulative sums of e**2, e**4, used to efficiently calculate var of e**2
        sum_E_squared = np.cumsum(E**2)                 # !!! Calculating the cumulative sums like this will lead to floating point precision errors !!!
                                                        # However they seem to be negliable for reasonable trace sizes < 1.000.000. Should be safe even for much larger traces till 100.000.000 datapoints and beyond
        
        CF = np.zeros_like(self.trace)                  # Charateristic function is computed dynamically (i.e. calculation is dependend on its past values)
                                                        # in the for loop and is therefore only initialized here
        cumultative_mean = np.zeros_like(self.trace)    # Mean of e**4 of the sample from the beginning of the trace up to the present evaluation point
        dynamic_variance = np.zeros_like(self.trace)    # Dynamically computed variance of e**2 on the same sample, but which is additionaly frozen in place
                                                        # if the charateristic function exceeds 2*S1
        pick_flag = None
        freeze_denom_calculation = False
        
        
        for i in range(len(self.trace)):
            n = i+1 # Current sample size
            
            # Compute the charateristic function
            cumultative_mean[i] = sum_E_squared[i]/n                            # Mean of e**4 from [0,present data point]
            if freeze_denom_calculation:                                        # Variance is not updated if the CF has increased above 2*S1 in the iteration before
                dynamic_variance[i] = dynamic_variance[i-1]
            else:                                                               # Otherwise it is newly calculated after see Baer & Kradolfer 1987
                dynamic_variance[i] = sum_E_squared[i]/n - (sum_E[i]/n)**2      # Variance of e**2 = E on the samples of [0, present data point]
            
            CF[i] = np.abs((E[i]**2 - cumultative_mean[i])/dynamic_variance[i]) # Characterstic function after (Bear & Kradolfer 1987) and (Küperkoch et. al. 2012)
            
            if CF[i] > 2*S1:                                                    # Decide, whether to freeze the denominator calculation on the next iteration
                freeze_denom_calculation = True
            else:
                freeze_denom_calculation = False
            
            # If no pick flag is currently set, check whether the CF has increased above the treshold S1 and a new pick flag should be triggered
            if pick_flag is None:
                if CF[i] >= S1:
                    pick_flag = i
                    evaluated_window_length = 0
                    below_threshold_counter = 0
            
            # If a pick flag is currently set, evaluate the ongoing event
            else:
                # Update counters
                evaluated_window_length += 1        # present length of ongoing event
                if CF[i] < S1:
                    below_threshold_counter += 1    # length of which CF was below the threshold
                
                # An event is reject as soon as the CF cumulatively was below the threshold for longer than TDown
                if below_threshold_counter >= window_length_grace_period:
                    pick_flag = None
                else:
                    # An event is accepted as a genuine P-Phase pick if the prevails longer than TUp without the CF falling below S1 for longer than TDown
                    if evaluated_window_length >= window_length_to_evaluate:
                        # Return the pick represeted by its index and possibly return additional info about all of the computed variables.
                        self.pick_BK = pick_flag
                        if return_insights:
                            self.insights_BK = (CF, E, cumultative_mean, dynamic_variance, params)
                        else: # Delete insights from an old run of the routine, so there is no confusion
                            try:
                                delattr(self, 'insights_BK')
                            except AttributeError:
                                pass
                        ret = (pick_flag, (CF, E, cumultative_mean, dynamic_variance, params)) if return_insights else pick_flag
                        return ret
        
        # If no pick flag was succesfully accepted return None and possibly additional info about all of the computed variables.
        self.pick_BK = None
        if return_insights:
            self.insights_BK = (CF, E, cumultative_mean, dynamic_variance, params)
        else: # Delete insights from an old run of the routine, so there is no confusion
            try:
                delattr(self, 'insights_BK')
            except AttributeError:
                pass
        ret = (None, (CF, E, cumultative_mean, dynamic_variance, params)) if return_insights else None
        return ret
    
    #-------------------------------------------------------------------------#
    #                HELPER METHODS FOR PICKING ALGORITHMS                    #
    #-------------------------------------------------------------------------#
    def get_discrete_derivative(self):
        """
        Helper method, that calculates a balanced numerical derivative for a discrete time series
        with equal sample spacing. On the boundarys a normal forward/backward numerical derivative is used.
        """
        x = self.trace
        x_fwdshift = np.empty_like(x)
        x_fwdshift[0] = 2*x[0]-x[1]
        x_fwdshift[1:] = x[:-1]
        x_bwdshift = np.empty_like(x)
        x_bwdshift[-1] = 2*x[-1]-x[-2]
        x_bwdshift[:-1] = x[1:]
        return (x_bwdshift-x_fwdshift)/(2*self.dt) # (x_{i+1}-x_{i-1})/2*sample interval
    
    def calc_BK_envelope(self):
        """
        Helper method, that calculates the estimation of the squared envelope e**2 of the trace
        as described in (Baer & Kradolfer 1987).
        """
        x = self.trace
        delx = self.get_discrete_derivative()
        return x**2 + (np.cumsum(x**2)/np.cumsum(delx**2))*(delx**2) # see (Baer & Kradolfer 1987) 
    
    def calc_Allen_envelope(self):
        """
        Helper method, that caluclates the estimation of the squared envolope e**2 of the trace
        for the Allen picker. However the relative weight of the derivative is not a fixed constant
        as in (Allen 1978) but is computed as described in (Küperkoch et. al. 2012)
        """
        x = self.trace
        delx = self.get_discrete_derivative()
        x_fwdshift = np.empty_like(x)
        x_fwdshift[0] = 0
        x_fwdshift[1:] = x[:-1]
        x_differences = x-x_fwdshift
        return x**2 + np.cumsum(np.abs(x))/np.cumsum(np.abs(x_differences))*delx**2 # see (Küperkoch et. al. 2012)
    
    @staticmethod  
    def STA(x,sampling_interval,alpha,weights='exp'):
        """
        Helper method that calculates a uniformly weighted moving average of window length
        alpha or an exponentially weighted average, with the weight parameter representing
        a window of length alpha

        Parameters
        ----------
        x : numpy.ndarray
            Input time series data.
        sampling_interval : int or float
            Regular sampling interval in seconds [s].
        alpha : int or float
            Desired window length in seconds [s].
        weights : str, optional
            'uni' for a normal moving average with window size alpha.
            'exp' for a exponentially weighted moving average as in (Allen 1978).
            The weighting constant used to calculate the exponential weights, will
            then be chosen in a way to represent window length alpha, since the
            theoretical window lenght is infinte. (see https://en.wikipedia.org/wiki/Exponential_smoothing)
            The default is 'exp'.

        Returns
        -------
        sta : numpy.ndarray
            Averaged time series data.

        """
        x = np.nan_to_num(x)
        sta = np.zeros_like(x,dtype=float)
        window_length = int(np.ceil(alpha/sampling_interval))   # Index of window length
        sta[:window_length-1] = np.nan                          # Results are only returned after one complete window length
        if weights=='uni':
            # uniformly weighted moving average is equivalent to a convolution ot the time series with a 1/N box as the kernel
            # np.convolve is significantly faster for large traces size than using explicit list comprehensions or even for loops
            sta[window_length-1:] = np.convolve(x,np.ones(window_length)/window_length,mode='valid')
        elif weights=='exp':
            C3 = float(2/(window_length+1))                                     # Exponential weighting parameter C3 as in (Allen 1978)
            # Only use geometric weights, till they sum to 99.99% of the cumulative geometric distribution to save computing power
            cutoff_index = np.minimum(len(x),int(np.ceil(np.log(0.0001)/np.log(1-C3)-1))) 
            geometric_weights = C3*np.power((1-C3),np.arange(cutoff_index))     # Exponential weights follow a geometric distribution with parameter C3
            # EWMA is again equivalent to a convolution (np.convolve is faster than other methods)
            sta[window_length-1:] = np.convolve(x,geometric_weights,mode='full')[window_length-1:len(x)]
        else:
            raise ValueError("Please specify whether to use an uniformly or exponentially weighted average using weights = 'uni' or 'exp'")
        return sta
    
    @staticmethod
    def LTA(x,sampling_interval,beta,weights='exp'):
        """
        Helper method that calculates a uniformly weighted moving average of window length
        beta or an exponentially weighted average, with the weight parameter representing
        a window of length beta. A copy of STA with different variable naming.

        Parameters
        ----------
        x : numpy.ndarray
            Input time series data.
        sampling_interval : int or float
            Regular sampling interval in seconds [s].
        beta : int or float
            Desired window length in seconds [s].
        weights : str, optional
            'uni' for a normal moving average with window size beta.
            'exp' for a exponentially weighted moving average as in (Allen 1978).
            The weighting constant used to calculate the exponential weights, will
            then be chosen in a way to represent window length beta, since the
            theoretical window lenght is infinte. (see https://en.wikipedia.org/wiki/Exponential_smoothing)
            The default is 'exp'.

        Returns
        -------
        lta : numpy.ndarray
            Averaged time series data.

        """
        x = np.nan_to_num(x)
        lta = np.zeros_like(x,dtype=float)
        window_length = int(np.ceil(beta/sampling_interval))    # Index of window length
        lta[:window_length-1] = np.nan                          # Results are only returned after one complete window length
        if weights=='uni':
            # uniformly weighted moving average is equivalent to a convolution ot the time series with a 1/N box as the kernel
            # np.convolve is significantly faster for large traces size than using explicit list comprehensions or even for loops
            lta[window_length-1:] = np.convolve(x,np.ones(window_length)/window_length,mode='valid')
        elif weights=='exp':
            C4 = float(2/(window_length+1))                                     # Exponential weighting parameter C3 as in (Allen 1978)
            # Only use geometric weights, till they sum to 99.99% of the cumulative geometric distribution to save computing power
            cutoff_index = np.minimum(len(x),int(np.ceil(np.log(0.0001)/np.log(1-C4)-1))) 
            geometric_weights = C4*np.power((1-C4),np.arange(cutoff_index))     # Exponential weights follow a geometric distribution with parameter C3
            # EWMA is again equivalent to a convolution (np.convolve is faster than other methods)
            lta[window_length-1:] = np.convolve(x,geometric_weights,mode='full')[window_length-1:len(x)]
        else:
            raise ValueError("Please specify whether to use an uniformly or exponentially weighted average using weights = 'uni' or 'exp'")
        return lta
    
    
    def visualize_pick_info(self,event_name=''):
        """
        Method to visualize seismic traces and picks and additional information
        about the algorithms run by the Picker instance. The layout and stlye of
        the plots is hardcoded and there is no interface method to easily make modifications.
        The plotting function are in the module 'drawpicker.py' and code changes
        can be made there if necessary.

        Parameters
        ----------
        event_name : str, optional
            Additional string to be shown in the title of the diagram. The default is ''.

        Returns
        -------
        fig, ax: matplotlib.figure and matplotlib.axis instances or a list of such
            Contains the displayed figure and axis instances to be e.g. further modified or saved by the user

        """
        # Plotting parameters for matplotlib
        mpl.rcParams['figure.dpi'] = 300
        plt.rcParams['text.usetex'] = False
        # Construct shared time axis
        self.time_axis = np.arange(len(self.trace))*self.dt
        # Extract label text out of header data stored in the instance variable as obspy generated header dictonary
        trace_figtitle = ' '.join([self.header['network'],self.header['station'],self.header['location'],event_name])
        trace_channelname = self.header['channel']
        trace_utcstarttime = op.core.UTCDateTime(self.header['starttime'])
        trace_startdate = '.'.join([f"{trace_utcstarttime.day:02d}",f"{trace_utcstarttime.month:02d}",f"{trace_utcstarttime.year:04d}"])
        trace_starttime = ':'.join([f"{trace_utcstarttime.hour:02d}",f"{trace_utcstarttime.minute:02d}",f"{trace_utcstarttime.second:02d}"])
        
        # Figure if no additional information about the picker is available (i.e. insight instance variable have not been initialized by the picking routines)
        # Only shows the trace data and the picks if the picking routines have been run. Otherwise only plots the trace data
        if (not hasattr(self, 'insights_Allen')) and (not hasattr(self, 'insights_BK')):
            fig,ax = plt.subplots(figsize=[6.4,4.8])
            
            fig,ax = dp.draw_trace(self,fig,ax,trace_figtitle,trace_channelname)
            
            ax.set_xlim(self.time_axis[0],self.time_axis[-1])
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.set_xlabel('Time [s] since '+trace_starttime+', '+trace_startdate)
            
            plt.show()
            return fig,ax
        
        # Figure if the Allen Picker was run with insights to be returned. These are visualized additionally to the trace data and the picks
        elif hasattr(self, 'insights_Allen') and (not hasattr(self, 'insights_BK')):
            fig,ax = plt.subplots(4,1,sharex=True,figsize=[6.4,10])
            
            fig, ax[0] = dp.draw_trace(self, fig, ax[0], trace_figtitle+' Insights Allen', trace_channelname)
            ax[0].tick_params(labelbottom=True)
            
            CF, E, sta, lta, delta, M, L, s, params = self.insights_Allen
            alpha, beta, gamma, tMin, MMin = params
            
            fig, ax[1] = dp.draw_AllenCF(self, fig, ax[1], CF, E, alpha, beta, gamma)
            ax[1].tick_params(labelbottom=True)
            
            fig,ax[2] = dp.draw_AllenSTA(self, fig, ax[2], sta, lta, delta)
            ax[2].tick_params(labelbottom=True)
            
            fig, ax[3] = dp.draw_AllenM(self, fig, ax[3], M, L, s, tMin, MMin)
            
            ax[3].set_xlabel('Time [s] since '+trace_starttime+', '+trace_startdate)
            ax[3].set_xlim(self.time_axis[0],self.time_axis[-1])
            ax[3].xaxis.set_minor_locator(MultipleLocator(5))
            
            plt.show()
            return fig,ax
        
        # Figure if the BK Picker was run with insights to be returned. These are visualized additionally to the trace data and the picks
        elif (not hasattr(self, 'insights_Allen')) and hasattr(self, 'insights_BK'):
            fig,ax = plt.subplots(3,1,sharex=True,figsize=[6.4,7.35])
            
            fig, ax[0] = dp.draw_trace(self, fig, ax[0], trace_figtitle+' Insights BK', trace_channelname)
            ax[0].tick_params(labelbottom=True)
            
            CF, E, cumultative_mean, dynamic_variance, params = self.insights_BK
            TUp, TDown, S1 = params
            
            fig, ax[1] = dp.draw_BKCF(self, fig, ax[1], CF, TUp, TDown, S1)
            ax[1].tick_params(labelbottom=True)
            
            fig, ax[2] = dp.draw_BKE(self, fig, ax[2], E, cumultative_mean, dynamic_variance)
            
            ax[2].set_xlabel('Time [s] since '+trace_starttime+', '+trace_startdate)
            ax[2].set_xlim(self.time_axis[0],self.time_axis[-1])
            ax[2].xaxis.set_minor_locator(MultipleLocator(5))
            
            plt.show()
            return fig,ax
        
        # Plots both of the above figures if both picking routines were run to return further insights
        elif  hasattr(self, 'insights_Allen') and hasattr(self, 'insights_BK'):
            fig1,ax1 = plt.subplots(4,1,sharex=True,figsize=[6.4,10])
            
            fig1, ax1[0] = dp.draw_trace(self, fig1, ax1[0], trace_figtitle+' Insights Allen', trace_channelname)
            ax1[0].tick_params(labelbottom=True)
            
            CF, E, sta, lta, delta, M, L, s, params1 = self.insights_Allen
            alpha, beta, gamma, tMin, MMin = params1
            
            fig1, ax1[1] = dp.draw_AllenCF(self, fig1, ax1[1], CF, E, alpha, beta, gamma)
            ax1[1].tick_params(labelbottom=True)
            
            fig1,ax1[2] = dp.draw_AllenSTA(self, fig1, ax1[2], sta, lta, delta)
            ax1[2].tick_params(labelbottom=True)
            
            fig1, ax1[3] = dp.draw_AllenM(self, fig1, ax1[3], M, L, s, tMin, MMin)
            
            ax1[3].set_xlabel('Time [s] since '+trace_starttime+', '+trace_startdate)
            ax1[3].set_xlim(self.time_axis[0],self.time_axis[-1])
            ax1[3].xaxis.set_minor_locator(MultipleLocator(5))
            
            plt.show()
            
            fig2,ax2 = plt.subplots(3,1,sharex=True,figsize=[6.4,7.35])
            
            fig2, ax2[0] = dp.draw_trace(self, fig2, ax2[0], trace_figtitle+' Insights BK', trace_channelname)
            ax2[0].tick_params(labelbottom=True)
            
            CF, E, cumultative_mean, dynamic_variance, params = self.insights_BK
            TUp, TDown, S1 = params
            
            fig2, ax2[1] = dp.draw_BKCF(self, fig2, ax2[1], CF, TUp, TDown, S1)
            ax2[1].tick_params(labelbottom=True)
            
            fig2, ax2[2] = dp.draw_BKE(self, fig2, ax2[2], E, cumultative_mean, dynamic_variance)
            
            ax2[2].set_xlabel('Time [s] since '+trace_starttime+', '+trace_startdate)
            ax2[2].set_xlim(self.time_axis[0],self.time_axis[-1])
            ax2[2].xaxis.set_minor_locator(MultipleLocator(5))
            
            plt.show()
            
            return [fig1,fig2],[ax1,ax2]
        