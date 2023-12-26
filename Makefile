all: example

example:
	clang++ -std=c++17 -O0 -g -Wall main.cpp -o example
clean:
	rm -rf example
