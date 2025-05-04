There are a couple of places I need to change to make it work in 2025:
1. np.int, np.bool are deprecated, use int and bool instead
2. import moviepy.editor as mpy is deprecated, just change it to from moviepy import * 
3. MountainCar env's rendering is largely deprecated by gymnasium. Just inherit directly from gymnasium and only overwrite if necessary. Check mountain_hill_env.py
4. matplotlib's 2D linear set_data does not support scalar, just convert to list of scalar. Check double_integrator_env.py
