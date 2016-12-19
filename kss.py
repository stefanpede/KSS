import os
import pandas as pd
import numpy as np
import h5py
from functools import reduce
import collections
import matplotlib.pyplot as plt
import random
from shutil import copyfile
import sys


class CurveManager:
    def __init__(self,
                 curves_path,
                 dmvt_path,
                 attrs=('DPRD_ID', 'DPRD_PSEUDO_NR', 'DWLD_LEITGUETE', 'DWLD_WAND_SOLL',
                        'DWLD_AD_SOLL', 'DWLD_WALZLOS_NR', 'DWLD_ID'),
                 index=None,
                 h5py_kwargs=None
                 ):
        if h5py_kwargs is None:
            h5py_kwargs = {}

        # Safe locations
        self._dmvt_path = dmvt_path
        self._curves_path = curves_path
        self._attrs = attrs
        self._h5py_kwargs = h5py_kwargs

        # Check weather Files exist
        if not os.path.isfile(curves_path):
            # Create h5py File object
            self.curves_h5 = h5py.File(curves_path, **h5py_kwargs)

            # Check if dmvt_File esists:
            if os.path.isfile(dmvt_path):
                with pd.HDFStore(dmvt_path) as store:
                    self.dmvt = store['dmvt']
            else:
                self.dmvt = pd.DataFrame(data=None,
                                         columns=['Device', 'PPDB_Name', 'Kurzbeschreibung',
                                                  'Beschreibung', 'Einheit'],
                                         index=pd.Int64Index(data=(), name='DMVT_ID')
                                         )
                with pd.HDFStore(dmvt_path) as store:
                    store['dmvt'] = self.dmvt

        else:
            self.curves_h5 = h5py.File(curves_path, **h5py_kwargs)
            # Load panda Frames
            with pd.HDFStore(dmvt_path) as store:
                self.dmvt = store['dmvt']

        # Generate metadata from curves
        if index is None:
            self.index = self.generate_index(self.curves_h5)
        else:
            self.index = index

    def regenerate_index(self):
        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'
        self.index = self.generate_index(self.curves_h5)

    def save(self, dm_path=None, dmvt_path=None, curves_path=None):
        if dmvt_path is None:
            dmvt_path = self._dmvt_path

        if curves_path is not None and curves_path != self._curves_path:
            self.copy_datafile(curves_path)

        store = pd.HDFStore(dmvt_path)
        store['dmvt'] = self.dmvt
        store.close()

        if dm_path is not None:
            store = pd.HDFStore(dm_path)
            store['index'] = self.index
            store['paths'] = pd.Series([self._curves_path, self._dmvt_path])
            store['attrs'] = pd.Series([self._attrs])
            if self._h5py_kwargs:
                keys, values = zip(*self._h5py_kwargs.items())
            else:
                keys, values = (None, None)
            store['h5py_keys'] = pd.Series(keys)
            store['h5py_values'] = pd.Series(values)
            store.close()

    @staticmethod
    def load(dm_path):
        with pd.HDFStore(dm_path) as store:
            index = store['index']
            paths = store['paths']
            attrs = store['attrs'][0]
            keys = store['h5py_keys']
            values = store['h5py_values']
            h5py_kwargs = {k: v for k, v in zip(keys, values)}

        curves_path = paths[0]
        dmvt_path = paths[1]

        return CurveManager(curves_path, dmvt_path,
                            attrs=attrs, index=index, h5py_kwargs=h5py_kwargs)

    def copy_datafile(self, filename, structure_only=True):
        if os.path.normpath(filename) != os.path.normpath(self._curves_path):
            if structure_only:
                self.copy_structure(self._curves_path, filename)
            else:
                copyfile(self._curves_path, filename)

    @staticmethod
    def copy_structure(old_file, new_file):
        with h5py.File(old_file, 'r') as f_old:
            with h5py.File(new_file, 'w') as f_new:
                groups = f_old.items()
                total = len(groups)
                for i, (name, group) in enumerate(groups):
                    print_progress(i, total, prefix='Creating HDF5, Progress:', suffix='Complete')
                    g = f_new.create_group(name)
                    for att, att_val in group.attrs.items():
                        g.attrs[att] = att_val

    def select_tubes_from_index(self, query, columns=()):
        if columns == 'ALL':
            columns = self.index.columns

        if isinstance(query, dict):
            masks = [function(self.index[column]) for column, function in query.items()]
            mask = reduce(lambda x, y: x & y, masks)
            selection = self.index.loc[mask, columns]
        elif isinstance(query, str):
            selection = self.index.query(query).loc[:, columns]
        else:
            selection = self.index.loc[query, columns]
        return selection

    @staticmethod
    def generate_index(storage,
                       attrs=('DPRD_ID', 'DPRD_PSEUDO_NR', 'DWLD_LEITGUETE', 'DWLD_WAND_SOLL',
                              'DWLD_AD_SOLL', 'DWLD_WALZLOS_NR', 'DWLD_ID')
                       ):

        tubes = storage.items()

        # If no tubes exist in storage return empty index
        if len(tubes) == 0:
            df = pd.DataFrame(data=None,
                              index=pd.Int64Index(data=(), name='DPRD_ID'),
                              columns=tuple([a for a in attrs if a != 'DPRD_ID']))
            return df

        # If tubes exist, find out which fields exist (dmvt_ids) and add the corresponding lengths
        # to the index.
        _, data = next(tubes.__iter__())
        curves = tuple(data.keys())
        total = len(tubes)
        meta = []
        for i, (_, data) in enumerate(tubes):
            print_progress(i, total, prefix='Generating Index, Progress:', suffix='Complete')
            # Alternative: df.loc[i] = ...   This is however MUCH MUCH slower.
            meta.append([data.attrs[a] for a in attrs] +
                        [0 if (data[c].shape == () and np.isnan(data[c]))
                         else 1 if (data[c].shape == ())
                        else len(data[c])
                         for c in curves])

        df = pd.DataFrame(meta, columns=attrs + curves)
        df = df.set_index('DPRD_ID')
        return df

    def load_curves(self, dmvt, selection):
        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'
        selection = self._selection2int(selection)
        selection = selection.reshape((-1,))
        dmvt_ids = self.dmvt2str(dmvt)

        if isinstance(dmvt_ids, str):
            single_dmvt_flag = True
        else:
            single_dmvt_flag = False
        valid_pattern = np.in1d(selection, self.index.index)
        valid_dprd_ids = selection[valid_pattern]
        for dprd_id in valid_dprd_ids:
            if single_dmvt_flag:
                curves = np.array(self.curves_h5[str(dprd_id) + '/' + dmvt_ids])
            else:
                curves = [np.array(self.curves_h5[str(dprd_id) + '/' + dmvt_id]) for
                          dmvt_id in dmvt_ids]
            yield dprd_id, curves

    def _selection2boolean(self, selection):
        """ Returns boolean iterable with length equal to len(self.index) """
        if isinstance(selection, (collections.Sequence, np.ndarray)) and \
                        np.array(selection).dtype == np.dtype('bool'):
            if len(selection) != len(self.index):
                raise ValueError('Boolean array "selection" does not match index dimension')
            else:
                return selection
        elif isinstance(selection, (collections.Sequence, np.ndarray)):
            return self.index.index.isin(selection)
        else:
            return self.index.index == selection

    def _selection2int(self, selection):
        """ Returns DPRD_IDs in a numpy array"""
        if isinstance(selection, (collections.Sequence, np.ndarray)) and \
                        np.array(selection).dtype == np.dtype('bool'):
            if len(selection) != len(self.index):
                raise ValueError('Boolean array "selection" does not match index dimension')
            else:
                return np.array(self.index.index[selection]).squeeze()
        elif isinstance(selection, pd.DataFrame):
            return np.array(selection.index).squeeze()
        elif isinstance(selection, (collections.Sequence, np.ndarray)):
            return np.array(selection).squeeze()
        else:
            return np.array([selection]).squeeze()

    def dmvt2int(self, dmvt):
        if isinstance(dmvt, (np.int, np.int_)):
            assert (dmvt in self.dmvt.index), 'DMVT-ID {} does not exist!'.format(dmvt)
            return dmvt
        elif isinstance(dmvt, str):
            assert (dmvt in self.dmvt['Kurzbeschreibung'].values), \
                'DMVT-ID {} does not exist!'.format(dmvt)
            d = self.dmvt[self.dmvt['Kurzbeschreibung'] == dmvt].index[0]
            return d
        elif (isinstance(dmvt, (collections.Sequence, np.ndarray))) and not isinstance(dmvt[0],
                                                                                       str):
            # Case: Iterable of integers or floats
            dmvt = np.array(dmvt)
            assert all([d in self.dmvt.index for d in dmvt]), \
                'DMVT-ID {} does not exist!'.format([d for d in dmvt if d not in self.dmvt.index])
            return dmvt
        else:
            # Case: Iterable of strings
            assert all([d in self.dmvt['Kurzbeschreibung'].values for d in dmvt]), \
                'DMVT-ID {} does not exist!'.format(
                    [d for d in dmvt if d not in self.dmvt['Kurzbeschreibung'].values])
            dmvt = np.array([self.dmvt[self.dmvt['Kurzbeschreibung'] == item].index[0]
                             for item in dmvt], dtype=np.int)
            return dmvt

    def dmvt2str(self, dmvt):
        dmvt_int = self.dmvt2int(dmvt)
        if not isinstance(dmvt_int, (collections.Sequence, np.ndarray)):
            return self.dmvt.at[dmvt_int, 'Kurzbeschreibung']
        else:
            return np.array(self.dmvt.loc[dmvt_int, 'Kurzbeschreibung'])

    def plot_dmvt_id(self, selection, dmvt, show_legend=False, figsize=(13, 5),
                     axes_kwargs=None):
        if axes_kwargs is None:
            axes_kwargs = {}
        dmvt = self.dmvt2int(dmvt)
        if isinstance(dmvt, (collections.Sequence, np.ndarray)):
            num_plots = len(dmvt)
        else:
            num_plots = 1

        fig, axes = plt.subplots(num_plots, 1, figsize=figsize)

        if num_plots == 1:
            # Make objects iterable
            axes = [axes]
            dmvt = [dmvt]

        for ax, cur_dmvt in zip(axes, dmvt):
            for dprd_id, curve in self.load_curves(cur_dmvt, selection):
                ax.plot(curve, label=dprd_id)
            ax.set(xlabel='Segment / n',
                   ylabel=self.dmvt2str(cur_dmvt),
                   **axes_kwargs)
            if show_legend:
                ax.legend(loc=2)

    def plot_random(self, amount, dmvt_id, selection='ALL', show_legend=False, figsize=(13,5),
                    axes_kwargs=None):
        if axes_kwargs is None:
            axes_kwargs = {}
        if selection == 'ALL':
            selection = self.index.index
        selection = self._selection2int(selection)
        plot_idx = random.sample(range(len(selection)), min(len(selection), amount))
        self.plot_dmvt_id(selection[plot_idx], dmvt_id, show_legend,
                          figsize=figsize, axes_kwargs=axes_kwargs)

    def add_dmvt(self,
                 identifier,
                 short_description,
                 device=np.nan,
                 ppdb_name=np.nan,
                 description=np.nan,
                 unit=np.nan,
                 create_hdf5_groups=True):

        assert isinstance(short_description, str), 'Short description must be of type String!'
        assert isinstance(identifier, (np.int, np.int_)), 'Identifier must be of type Int!'
        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'

        # Add new ID to dmvt-Table
        self.dmvt.loc[identifier, :] = (device, ppdb_name, short_description, description, unit)

        # Create new column in index
        if len(self.index.index) == 0:
            self.index[short_description] = ''
        else:
            self.index.loc[:, short_description] = 0

        # Add new ID to HDF5-File
        if create_hdf5_groups:
            tubes = self.curves_h5.items()
            total = len(tubes)
            for i, (_, tube) in enumerate(tubes):
                print_progress(i, total, prefix='Add DMVT to HDF5, Progress:', suffix='Complete')
                tube.create_dataset(short_description, data=np.nan)

    def remove_dmvt(self, dmvt_int, short_description):
        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'

        # Remove from index
        if short_description in self.index.columns:
            self.index.drop(short_description, axis=1, inplace=True)

        # Remove from HDF5
        tubes = self.curves_h5.items()
        total = len(tubes)
        for i, (_, tube) in enumerate(tubes):
            print_progress(i, total, prefix='Remove DMVT from HDF5, Progress:', suffix='Complete')
            if short_description in list(tube.keys()):
                del tube[short_description]

        # Remove from dmvt-Table
        if dmvt_int in self.dmvt.index:
            self.dmvt.drop(dmvt_int, axis=0, inplace=True)

    def _overwrite_curve_in_hdf5(self, dmvt_str, dprd_id, curve):
        p = str(dprd_id) + '/' + dmvt_str
        del self.curves_h5[p]
        self.curves_h5[str(dprd_id)].create_dataset(dmvt_str, data=curve)

    def overwrite_curve(self, dmvt, dprd_id, curve):
        """ Stores given curve in HDF5-File and updates Index. The given DPRD-ID and DMVT-ID must
        already exit.
        """
        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'

        # Cast dmvt to string. This throws an error if the dmvt does not exist
        dmvt = self.dmvt2str(dmvt)

        # Check if the dprd_id exists
        assert dprd_id in self.index.index, 'DPRD_ID is not in index. Please use the add_dprd_id ' \
                                            'function to add a new tube!'

        # Add in HDF5
        curve = np.array(curve)
        self._overwrite_curve_in_hdf5(dmvt, dprd_id, curve)
        if not (curve.shape == () and np.isnan(curve)):
            # Add length of curve to index
            self.index.at[dprd_id, dmvt] = len(curve) if curve.shape != () else 1
        else:
            # For nan length is 0
            self.index.at[dprd_id, dmvt] = 0

    def fast_add_dprd_id(self, attrs: dict, curves: dict, existing: bool):
        """ USE WITH CAUTION:
        Assumes that the datasets to be created do not yet exist.
        The dprd_ids existance is indicated with 'existing'.

        The Index will not be updated!
        """

        dprd_id = int(attrs['DPRD_ID'])

        if existing:
            hdf5_tube = self.curves_h5[str(dprd_id)]
            for key, curve in curves.items():
                curve = np.asarray(curve)
                hdf5_tube.create_dataset(key, data=curve)

        else:
            hdf5_tube = self.curves_h5.create_group(str(dprd_id))
            # Add Attributes
            for key, val in attrs.items():
                hdf5_tube.attrs[key] = val

            # Add Curves
            for key, curve in curves.items():
                curve = np.asarray(curve)
                hdf5_tube.create_dataset(key, data=curve)

    def batch_add_dprd_ids(self, attrs: list, curves, dmvt_ids: list):
        """
        curves: Iterator that yields a tuple of length len(dmvt_ids) at each iteration
        containing the corresponding curves to the DMVT-IDs.
        dmvt_ids: list dmvt_ids
        The Index will not be updated.
        After all batches are added the .regenerate_index() method should be called."""

        raise NotImplementedError

    def add_dprd_id(self, attrs: dict, curves: dict):
        """ Add curves to HDF5-File. If the DPRD-ID doesn't exist, it is created. If the dataset
        already exists it will be overwritten. The Index is updated correspondingly.
        This function takes care of all the overhead. It is however quite slow. Please consider
        using the batch_add_dprd_id function.

        Main performance issue: Reindexing of pandas DataFrame after each call of this function.
        """
        dprd_id = int(attrs['DPRD_ID'])

        assert self.curves_h5 is not None, 'HDF5-File closed. Please open HDF5-File before using ' \
                                           'it.'

        # Load first tube to get necessary fields
        c_fields = set(self.dmvt.loc[:, 'Kurzbeschreibung'])

        assert set(self._attrs) == set(attrs.keys()), 'Different set of attributes specified. ' \
                                                      'Necessary attributes: ' \
                                                      + ','.join(self._attrs)
        assert all([(k in c_fields) for k in curves.keys()]), \
            'DMVT_ID {} not found. Please use the add_dmvt function to add a new ID.'.format(
                [k for k in curves.keys() if k not in c_fields])

        # Check if tube already exists
        if dprd_id in self.index.index:
            hdf5_tube = self.curves_h5[str(dprd_id)]

            # Check if attributes are equal
            for a in self._attrs:
                assert attrs[a] == hdf5_tube.attrs[a], 'Attributes do not match for DPRD {}! \n' \
                                                       'HDF-File: {} : {}\n' \
                                                       'Arguments: {} : {}' \
                                                       ''.format(dprd_id,
                                                                 a, hdf5_tube.attrs[a],
                                                                 a, attrs[a])
            # Add in HDF5
            for key, curve in curves.items():
                self.overwrite_curve(dmvt=key, dprd_id=dprd_id, curve=curve)

        # If tube doesn't exist, create it
        else:
            # Ensure all necessary fields occur in the curves-dict.
            curves = {key: (curves[key] if key in curves else np.nan) for key in c_fields}

            # Prepare index entry
            for key, val in attrs.items():
                if key != 'DPRD_ID':
                    self.index.loc[dprd_id, key] = val

            hdf5_tube = self.curves_h5.create_group(str(dprd_id))

            # Add in HDF5

            # Add Attributes
            for key, val in attrs.items():
                hdf5_tube.attrs[key] = val

            # Add Curves
            for key, curve in curves.items():
                curve = np.asarray(curve)
                hdf5_tube.create_dataset(key, data=curve)

                if not (curve.shape == () and np.isnan(curve)):
                    # Add length of curve to index
                    self.index.at[dprd_id, key] = len(curve) if curve.shape != () else 1
                else:
                    self.index.at[dprd_id, key] = 0

            assert not any(self.index.loc[dprd_id, :].isnull()), 'Something went wrong -.-'

    def close_h5py(self):
        if self.curves_h5 is not None:
            self.curves_h5.close()
            self.curves_h5 = None

    def open_h5py(self, h5py_kwargs=None):
        if h5py_kwargs is None:
            h5py_kwargs = self._h5py_kwargs
        else:
            self._h5py_kwargs = h5py_kwargs
        if self.curves_h5 is None:
            self.curves_h5 = h5py.File(self._curves_path, **h5py_kwargs)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=70):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()