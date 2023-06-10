%  This script will attempt to compile the C++ code "pgdraw.cpp".
%
%  Requirements: 
%  C++ compiler that supports OpenMP (Visual Studio Professional or g++)


%% Determine how to compile for each operating system
compiler = mex.getCompilerConfigurations('C++','Selected'); 
fprintf('Compiling pgdraw.cpp ...\n');
if ispc()
    %% Windows with MS VisualStudio Pro
    ompdir = ['-L',fullfile(matlabroot,'bin','win64','')]; 
    mex(ompdir, 'COMPFLAGS=$COMPFLAGS /Ox /MD /openmp', 'LINKFLAGS=$LINKFLAGS /NODEFAULTLIB:vcomp', '-llibiomp5md', 'pgdraw.cpp');
   
elseif ismac()
    %% MAC with clang++
    %ompdir = ['-L',fullfile(matlabroot,'sys','os','maci64','')];    
    % -fopenmp=libomp
    %mex(ompdir, 'CXXFLAGS=$CXXFLAGS -O2 -std=c++11 -fopenmp', 'LDFLAGS=$LDFLAGS ', '-liomp5', 'pgdraw.cpp');
    error('This script does not work under the MacOSX operating system. Please go to http://www.emakalic.org/blog/ or http://dschmidt.org/?page_id=189 to download pre-compiled MEX files.');
    
else
    %% Linux with g++
    ompdir = ['-L',fullfile(matlabroot,'sys','os','glnxa64','')];
    mex(ompdir, 'CXXFLAGS=$CXXFLAGS -O2 -std=c++11 -fopenmp', 'LDFLAGS=$LDFLAGS ', '-liomp5', 'pgdraw.cpp');
    
end

