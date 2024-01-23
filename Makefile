all: example prof

example:
	clang++ -std=c++17 -O0 -g -Wall main.cpp -o example
prof:
	clang++ -std=c++17 -O0 -pg -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -Wall main.cpp -o prof
clean:
	rm -rf example
