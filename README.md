This started off with as a fork from [difPy](https://github.com/elisemercury/Duplicate-Image-Finder).

With the following major changes:

- the pixels values for every file is combined into a numpy matrix, including the rotations.
- used matrix broadcasting to compute the absolute difference (instead of MSE), to avoid/reduce looping
- Added memory constrains to the broadcasting size to avoid out of memory issues
- Changed the dependency from opencv to pillow and also used pathlibs instead of strs
- Also changed it to a stand alone function instead of a class and added option to move the duplicates found to a folder.
