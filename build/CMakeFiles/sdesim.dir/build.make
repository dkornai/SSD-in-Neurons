# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build

# Include any dependencies generated for this target.
include CMakeFiles/sdesim.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sdesim.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sdesim.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sdesim.dir/flags.make

CMakeFiles/sdesim.dir/sdesim.cpp.o: CMakeFiles/sdesim.dir/flags.make
CMakeFiles/sdesim.dir/sdesim.cpp.o: ../sdesim.cpp
CMakeFiles/sdesim.dir/sdesim.cpp.o: CMakeFiles/sdesim.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sdesim.dir/sdesim.cpp.o"
	/home/daniel/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sdesim.dir/sdesim.cpp.o -MF CMakeFiles/sdesim.dir/sdesim.cpp.o.d -o CMakeFiles/sdesim.dir/sdesim.cpp.o -c /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/sdesim.cpp

CMakeFiles/sdesim.dir/sdesim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sdesim.dir/sdesim.cpp.i"
	/home/daniel/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/sdesim.cpp > CMakeFiles/sdesim.dir/sdesim.cpp.i

CMakeFiles/sdesim.dir/sdesim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sdesim.dir/sdesim.cpp.s"
	/home/daniel/anaconda3/bin/x86_64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/sdesim.cpp -o CMakeFiles/sdesim.dir/sdesim.cpp.s

# Object files for target sdesim
sdesim_OBJECTS = \
"CMakeFiles/sdesim.dir/sdesim.cpp.o"

# External object files for target sdesim
sdesim_EXTERNAL_OBJECTS =

../libsdesim.so: CMakeFiles/sdesim.dir/sdesim.cpp.o
../libsdesim.so: CMakeFiles/sdesim.dir/build.make
../libsdesim.so: CMakeFiles/sdesim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../libsdesim.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sdesim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sdesim.dir/build: ../libsdesim.so
.PHONY : CMakeFiles/sdesim.dir/build

CMakeFiles/sdesim.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sdesim.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sdesim.dir/clean

CMakeFiles/sdesim.dir/depend:
	cd /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build /home/daniel/Projects/2023_Jones_Lab_mtDNA/SSD/SSD_Neuron/build/CMakeFiles/sdesim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sdesim.dir/depend

