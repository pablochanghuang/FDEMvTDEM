import os
import numpy as np


__all__ = [
    'anderson_801_1982',
    'gupt_61_1997',
    'gupt_120_1997',
    'gupt_47_1997',
    'gupt_140_1997',
    'kong_61_2007b',
    'kong_121_2007',
    'kong_241_2007',
    'key_101_2009',
    'key_201_2009',
    'key_401_2009',
    'key_51_2012',
    'key_101_2012',
    'key_201_2012',
    'wer_201_2018',
    'wer_2001_2018',
]

_LIBPATH = os.path.abspath(os.path.dirname(__file__))


def anderson_801_1982():
    """801 point Hankel filter, J0 and J1


    > Anderson, W. L., 1982;
    > Fast Hankel transforms using related and lagged convolutions;
    > ACM Trans. on Math. Softw. (TOMS), 8, 344-368;
    > DOI: 10.1145/356012.356014


    The original filter values are published in the appendix to above article:
    Algorithm 588, DOI: 10.1145/356012.356015

    The values provided here are taken from code that accompanies:
    Key 2012, Geophysics, DOI: 10.1190/geo2011-0237.1


    Copyright 1982 Walter L. Anderson

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(anderson_801_1982, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_anderson_801_1982_j0j1.npz'
        anderson_801_1982.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return anderson_801_1982.cache


anderson_801_1982.values = ['j0', 'j1']


def gupt_61_1997():
    """61 point Hankel filter, J0



    > Guptasarma, D. and B. Singh, 1997;
    > New digital linear filters for Hankel J0 and J1 transforms;
    > Geophysical Prospecting, 45(5), 745-762;
    > DOI: 10.1046/j.1365-2478.1997.500292.x


    Copyright 1997 D. Guptasarma and B. Singh

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0 : ndarray
        Filter base and its values.

    """
    if getattr(gupt_61_1997, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_gupt_61_1997_j0.npz'
        gupt_61_1997.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return gupt_61_1997.cache


gupt_61_1997.values = ['j0']


def gupt_120_1997():
    """120 point Hankel filter, J0



    > Guptasarma, D. and B. Singh, 1997;
    > New digital linear filters for Hankel J0 and J1 transforms;
    > Geophysical Prospecting, 45(5), 745-762;
    > DOI: 10.1046/j.1365-2478.1997.500292.x


    Copyright 1997 D. Guptasarma and B. Singh

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0 : ndarray
        Filter base and its values.

    """
    if getattr(gupt_120_1997, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_gupt_120_1997_j0.npz'
        gupt_120_1997.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return gupt_120_1997.cache


gupt_120_1997.values = ['j0']


def gupt_47_1997():
    """47 point Hankel filter, J1



    > Guptasarma, D. and B. Singh, 1997;
    > New digital linear filters for Hankel J0 and J1 transforms;
    > Geophysical Prospecting, 45(5), 745-762;
    > DOI: 10.1046/j.1365-2478.1997.500292.x


    Copyright 1997 D. Guptasarma and B. Singh

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j1 : ndarray
        Filter base and its values.

    """
    if getattr(gupt_47_1997, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_gupt_47_1997_j1.npz'
        gupt_47_1997.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return gupt_47_1997.cache


gupt_47_1997.values = ['j1']


def gupt_140_1997():
    """140 point Hankel filter, J1



    > Guptasarma, D. and B. Singh, 1997;
    > New digital linear filters for Hankel J0 and J1 transforms;
    > Geophysical Prospecting, 45(5), 745-762;
    > DOI: 10.1046/j.1365-2478.1997.500292.x


    Copyright 1997 D. Guptasarma and B. Singh

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j1 : ndarray
        Filter base and its values.

    """
    if getattr(gupt_140_1997, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_gupt_140_1997_j1.npz'
        gupt_140_1997.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return gupt_140_1997.cache


gupt_140_1997.values = ['j1']


def kong_61_2007b():
    """61 point Hankel filter, J0 and J1


    Designed and tested for dipole antenna radiation in a conductive medium.


    > Kong, F. N, 2007;
    > Hankel transform filters for dipole antenna radiation in a conductive
    > medium;
    > Geophysical Prospecting, 55(1), 83-89;
    > DOI: 10.1111/j.1365-2478.2006.00585.x


    These filter values are available from

      http://www.em-earth-consulting.no

    in the three files YBASE61NEW.dat, J0K61NEW.dat, and J1K61NEW.dat.
    Please consult the original source for more details.

    The appendix "b" after the year indicates that it corresponds to the NEW
    set of filter values.


    Copyright 2007 Fannian Kong

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(kong_61_2007b, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_kong_61_2007b_j0j1.npz'
        kong_61_2007b.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return kong_61_2007b.cache


kong_61_2007b.values = ['j0', 'j1']


def kong_121_2007():
    """121 point Hankel filter, J0 and J1


    Designed and tested for dipole antenna radiation in a conductive medium.


    > Kong, F. N, 2007;
    > Hankel transform filters for dipole antenna radiation in a conductive
    > medium;
    > Geophysical Prospecting, 55(1), 83-89;
    > DOI: 10.1111/j.1365-2478.2006.00585.x


    These filter values are available from

      http://www.em-earth-consulting.no

    in the three files YBASE121.dat, J0K121.dat, and J1K121.dat.
    Please consult the original source for more details.


    Copyright 2007 Fannian Kong

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(kong_121_2007, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_kong_121_2007_j0j1.npz'
        kong_121_2007.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return kong_121_2007.cache


kong_121_2007.values = ['j0', 'j1']


def kong_241_2007():
    """241 point Hankel filter, J0 and J1


    Designed and tested for dipole antenna radiation in a conductive medium.


    > Kong, F. N, 2007;
    > Hankel transform filters for dipole antenna radiation in a conductive
    > medium;
    > Geophysical Prospecting, 55(1), 83-89;
    > DOI: 10.1111/j.1365-2478.2006.00585.x


    These filter values are available from

      http://www.em-earth-consulting.no

    in the three files YBASE241.dat, J0K241.dat, and J1K241.dat.
    Please consult the original source for more details.


    Copyright 2007 Fannian Kong

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(kong_241_2007, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_kong_241_2007_j0j1.npz'
        kong_241_2007.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return kong_241_2007.cache


kong_241_2007.values = ['j0', 'j1']


def key_101_2009():
    """101 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2009;
    > 1D inversion of multicomponent, multifrequency marine CSEM data:
    > Methodology and synthetic studies for resolving thin resistive layers;
    > Geophysics, 74(2), F9-F20;
    > DOI: 10.1190/1.3058434


    Copyright 2009 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_101_2009, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_101_2009_j0j1.npz'
        key_101_2009.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_101_2009.cache


key_101_2009.values = ['j0', 'j1']


def key_201_2009():
    """201 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2009;
    > 1D inversion of multicomponent, multifrequency marine CSEM data:
    > Methodology and synthetic studies for resolving thin resistive layers;
    > Geophysics, 74(2), F9-F20;
    > DOI: 10.1190/1.3058434


    Copyright 2009 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_201_2009, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_201_2009_j0j1.npz'
        key_201_2009.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_201_2009.cache


key_201_2009.values = ['j0', 'j1']


def key_401_2009():
    """401 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2009;
    > 1D inversion of multicomponent, multifrequency marine CSEM data:
    > Methodology and synthetic studies for resolving thin resistive layers;
    > Geophysics, 74(2), F9-F20;
    > DOI: 10.1190/1.3058434


    Copyright 2009 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_401_2009, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_401_2009_j0j1.npz'
        key_401_2009.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_401_2009.cache


key_401_2009.values = ['j0', 'j1']


def key_51_2012():
    """51 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2012;
    > Is the fast Hankel transform faster than quadrature?;
    > Geophysics, 77(3), F21-F30;
    > DOI: 10.1190/geo2011-0237.1


    Copyright 2012 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_51_2012, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_51_2012_j0j1.npz'
        key_51_2012.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_51_2012.cache


key_51_2012.values = ['j0', 'j1']


def key_101_2012():
    """101 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2012;
    > Is the fast Hankel transform faster than quadrature?;
    > Geophysics, 77(3), F21-F30;
    > DOI: 10.1190/geo2011-0237.1


    Copyright 2012 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_101_2012, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_101_2012_j0j1.npz'
        key_101_2012.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_101_2012.cache


key_101_2012.values = ['j0', 'j1']


def key_201_2012():
    """201 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.


    > Key, K., 2012;
    > Is the fast Hankel transform faster than quadrature?;
    > Geophysics, 77(3), F21-F30;
    > DOI: 10.1190/geo2011-0237.1


    Copyright 2012 Kerry Key

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(key_201_2012, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_key_201_2012_j0j1.npz'
        key_201_2012.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return key_201_2012.cache


key_201_2012.values = ['j0', 'j1']


def wer_201_2018():
    """201 point Hankel filter, J0 and J1


    Designed and tested for controlled-source electromagnetic data.

    See the notebook `Filter-wer201.ipynb` in the repo
    https://github.com/emsig/article-fdesign


    > Werthmüller, D., K. Key, and E. Slob, 2019;
    > A tool for designing digital filters for the Hankel and Fourier
    > transforms in potential, diffusive, and wavefield modeling;
    > Geophysics, 84(2), F47-F56;
    > DOI: 10.1190/geo2018-0069.1


    Copyright 2018 Dieter Werthmüller

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(wer_201_2018, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_wer_201_2018_j0j1.npz'
        wer_201_2018.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return wer_201_2018.cache


wer_201_2018.values = ['j0', 'j1']


def wer_2001_2018():
    """2001 point Hankel filter, J0 and J1


    Designed and tested for ground-penetrating radar (GPR).

    See the notebook `Filter-wer2001.ipynb` in the repo
    https://github.com/emsig/article-fdesign


    > Werthmüller, D., K. Key, and E. Slob, 2019;
    > A tool for designing digital filters for the Hankel and Fourier
    > transforms in potential, diffusive, and wavefield modeling;
    > Geophysics, 84(2), F47-F56;
    > DOI: 10.1190/geo2018-0069.1


    Copyright 2018 Dieter Werthmüller

    This work is licensed under a CC BY 4.0 license.
    <http://creativecommons.org/licenses/by/4.0/>.


    Returns
    -------
    base, j0, j1 : ndarray
        Filter base and its values.

    """
    if getattr(wer_2001_2018, 'cache', None) is None:
        fname = 'lib/Hankel/hankel_wer_2001_2018_j0j1.npz'
        wer_2001_2018.cache = np.load(
            os.path.join(_LIBPATH, fname))['dlf']
    return wer_2001_2018.cache


wer_2001_2018.values = ['j0', 'j1']
