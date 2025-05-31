# Compile 
using the "Build Code" shortcut

# Benchmark all 4 memory types using
./memory_analysis.exe -m 256 -t 1024 -k global
./memory_analysis.exe -m 256 -t 1024 -k constant
./memory_analysis.exe -m 256 -t 1024 -k shared
./memory_analysis.exe -m 256 -t 1024 -k register

The results are saved in output csv.